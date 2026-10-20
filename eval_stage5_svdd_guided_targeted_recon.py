import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import utils as vutils
import pydicom
import numpy as np
from PIL import Image
import pandas as pd

from models.vqvae import VQVAE


# =============================
# Settings
# =============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

normal_data_dir = "/app/data/temp/normal"
abnormal_data_dir = "/app/data/temp/abnormal"

model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage1_full_final.pth"
stats_path = "/app/results/zq_stats_stage1_full.pth"

save_root = "/app/results/eval_stage5_svdd_guided_targeted_recon"

resize = 256
batch_size = 1
num_workers = 2

block_size = 48
fill_value = 0.08

# 只在 lung-like 区域找最大点，避免胸壁/身体区域干扰
use_lung_constraint = True


# =============================
# Model Parameters
# =============================

n_hiddens = 128
n_residual_hiddens = 64
n_residual_layers = 1

embedding_dim = 64
n_embeddings = 64
beta = 0.25


# =============================
# Dataset
# =============================

class DicomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, label):
        self.paths = []
        self.label = label

        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(".dcm"):
                    self.paths.append(os.path.join(root, f))

        self.paths = sorted(self.paths)
        print(f"[{label}] Found {len(self.paths)} DICOM files in {root_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        ds = pydicom.dcmread(path)
        image = ds.pixel_array.astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        image = image * slope + intercept

        window_min = -1000.0
        window_max = 400.0

        image = np.clip(image, window_min, window_max)
        image = (image - window_min) / (window_max - window_min)

        image = Image.fromarray((image * 255).astype(np.uint8))
        image = image.resize((resize, resize), Image.BILINEAR)
        image = np.array(image).astype(np.float32) / 255.0

        image = torch.tensor(image).unsqueeze(0)

        case_name = os.path.basename(path).replace(".dcm", "")

        return image, path, case_name


# =============================
# Load model
# =============================

model = VQVAE(
    n_hiddens,
    n_residual_hiddens,
    n_residual_layers,
    n_embeddings,
    embedding_dim,
    beta
).to(device)

checkpoint = torch.load(model_path, map_location=device)

if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

model.eval()

print("Loaded model:", model_path)


# =============================
# Load z_q stats
# =============================

stats = torch.load(stats_path, map_location=device)

center_map = stats["center_map"].to(device)
std_map = stats["std_map"].to(device)

print("Loaded stats:", stats_path)
print("center_map:", center_map.shape)
print("std_map:", std_map.shape)


# =============================
# Functions
# =============================

def normalize_map(x):
    x = x.detach()
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def extract_zq_and_recon(model, x):
    z_e = model.encoder(x)
    z_e = model.pre_quantization_conv(z_e)

    if hasattr(model, "pos_bias"):
        if model.pos_bias.shape[2] == z_e.shape[2] and model.pos_bias.shape[3] == z_e.shape[3]:
            z_e = z_e + model.pos_bias
        else:
            pos_bias = F.interpolate(
                model.pos_bias,
                size=z_e.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            z_e = z_e + pos_bias

    embedding_loss, z_q, perplexity, _, _, q_distance_map = model.vector_quantization(z_e)

    x_hat = model.decoder(z_q)

    return x_hat, z_q, q_distance_map, perplexity


def make_lung_like_mask(x):
    """
    x: [1,1,256,256]
    return: [1,1,256,256]
    """
    lung = ((x > 0.02) & (x < 0.45)).float()

    # 稍微扩张一点，包含胸膜边界
    lung = F.max_pool2d(
        lung,
        kernel_size=11,
        stride=1,
        padding=5
    )

    return lung


def get_svdd_map(z_q, center_map, std_map):
    """
    z_q: [B,64,64,64]
    return:
      svdd_map_64:  [B,1,64,64]
      svdd_map_256: [B,1,256,256]
    """
    zscore = ((z_q - center_map) / (std_map + 1e-6)) ** 2
    svdd_map_64 = zscore.mean(dim=1, keepdim=True)

    svdd_map_256 = F.interpolate(
        svdd_map_64,
        size=(256, 256),
        mode="bilinear",
        align_corners=False
    )

    return svdd_map_64, svdd_map_256


def find_best_mask_from_svdd(svdd_map_256, x):
    """
    svdd_map_256: [1,1,256,256]
    x: [1,1,256,256]
    return y1,y2,x1,x2,best_score
    """

    score = svdd_map_256.clone()

    if use_lung_constraint:
        lung_mask = make_lung_like_mask(x)
        score = score * lung_mask

    # 为了让 block 不超出边界，先找最大点
    flat_idx = torch.argmax(score.view(-1))
    cy = int(flat_idx // 256)
    cx = int(flat_idx % 256)

    half = block_size // 2

    y1 = max(0, cy - half)
    y2 = min(256, cy + half)

    x1 = max(0, cx - half)
    x2 = min(256, cx + half)

    # 如果靠边导致尺寸不够，补回 block_size
    if y2 - y1 < block_size:
        if y1 == 0:
            y2 = min(256, block_size)
        elif y2 == 256:
            y1 = max(0, 256 - block_size)

    if x2 - x1 < block_size:
        if x1 == 0:
            x2 = min(256, block_size)
        elif x2 == 256:
            x1 = max(0, 256 - block_size)

    best_score = score[:, :, y1:y2, x1:x2].mean().item()

    return y1, y2, x1, x2, best_score


def apply_targeted_mask(x, y1, y2, x1, x2):
    x_masked = x.clone()
    x_masked[:, :, y1:y2, x1:x2] = fill_value

    mask = torch.zeros_like(x)
    mask[:, :, y1:y2, x1:x2] = 1.0

    return x_masked, mask


def evaluate_dataset(data_dir, label):
    dataset = DicomDataset(data_dir, label)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    save_dir = os.path.join(save_root, label)
    os.makedirs(save_dir, exist_ok=True)

    records = []

    with torch.no_grad():
        for idx, (x, path, case_name) in enumerate(loader):
            x = x.to(device)
            case_name = case_name[0]
            path = path[0]

            # ordinary reconstruction
            recon_original, z_q, q_distance_map, perplexity = extract_zq_and_recon(model, x)
            diff_original = torch.abs(x - recon_original)

            # svdd map
            svdd_map_64, svdd_map_256 = get_svdd_map(
                z_q,
                center_map,
                std_map
            )

            # choose targeted region
            y1, y2, x1, x2, best_score = find_best_mask_from_svdd(
                svdd_map_256,
                x
            )

            # targeted mask
            x_masked, targeted_mask = apply_targeted_mask(
                x,
                y1,
                y2,
                x1,
                x2
            )

            # targeted reconstruction
            recon_targeted, z_q_targeted, q_distance_map_targeted, perplexity_targeted = extract_zq_and_recon(
                model,
                x_masked
            )

            diff_targeted = torch.abs(x - recon_targeted)

            # scores
            ordinary_diff_score = diff_original.mean().item()
            targeted_diff_score = diff_targeted.mean().item()

            targeted_region_diff_score = (
                diff_targeted * targeted_mask
            ).sum().item() / (targeted_mask.sum().item() + 1e-8)

            svdd_mean = svdd_map_256.mean().item()
            svdd_max = svdd_map_256.max().item()

            # qdistance upsample
            q_map = q_distance_map.unsqueeze(1)
            q_map = F.interpolate(
                q_map,
                size=(256, 256),
                mode="bilinear",
                align_corners=False
            )

            lung_mask = make_lung_like_mask(x)

            # save image
            grid = vutils.make_grid(
                torch.cat([
                    x.cpu(),
                    recon_original.cpu(),
                    normalize_map(diff_original).cpu(),
                    normalize_map(svdd_map_256).cpu(),
                    targeted_mask.cpu(),
                    x_masked.cpu(),
                    recon_targeted.cpu(),
                    normalize_map(diff_targeted).cpu(),
                    normalize_map(q_map).cpu(),
                    lung_mask.cpu(),
                ], dim=0),
                nrow=10,
                normalize=True
            )

            save_path = os.path.join(
                save_dir,
                f"{idx:04d}_{case_name}.png"
            )

            vutils.save_image(grid, save_path)

            records.append({
                "label": label,
                "case_name": case_name,
                "path": path,
                "y1": y1,
                "y2": y2,
                "x1": x1,
                "x2": x2,
                "svdd_best_score": best_score,
                "svdd_mean": svdd_mean,
                "svdd_max": svdd_max,
                "ordinary_diff": ordinary_diff_score,
                "targeted_diff": targeted_diff_score,
                "targeted_region_diff": targeted_region_diff_score,
                "perplexity_original": float(perplexity.item()),
                "perplexity_targeted": float(perplexity_targeted.item()),
                "image_path": save_path
            })

            print(
                f"[{label}] {idx+1}/{len(dataset)} "
                f"{case_name} "
                f"mask=({y1}:{y2},{x1}:{x2}) "
                f"svdd_best={best_score:.6f} "
                f"orig_diff={ordinary_diff_score:.6f} "
                f"target_diff={targeted_diff_score:.6f} "
                f"region_diff={targeted_region_diff_score:.6f}"
            )

    return records


# =============================
# Run
# =============================

os.makedirs(save_root, exist_ok=True)

all_records = []

all_records += evaluate_dataset(normal_data_dir, "normal")
all_records += evaluate_dataset(abnormal_data_dir, "abnormal")

df = pd.DataFrame(all_records)

csv_path = os.path.join(save_root, "stage5_svdd_guided_targeted_recon_scores.csv")
df.to_csv(csv_path, index=False)

print("Saved CSV:", csv_path)


# =============================
# Print summary
# =============================

for score_name in [
    "svdd_best_score",
    "svdd_mean",
    "svdd_max",
    "ordinary_diff",
    "targeted_diff",
    "targeted_region_diff"
]:
    print()
    print(f"[{score_name}]")

    normal_scores = df[df["label"] == "normal"][score_name].values
    abnormal_scores = df[df["label"] == "abnormal"][score_name].values

    print(
        f"Normal:   mean={normal_scores.mean():.6f}, "
        f"std={normal_scores.std():.6f}, "
        f"min={normal_scores.min():.6f}, "
        f"max={normal_scores.max():.6f}"
    )

    print(
        f"Abnormal: mean={abnormal_scores.mean():.6f}, "
        f"std={abnormal_scores.std():.6f}, "
        f"min={abnormal_scores.min():.6f}, "
        f"max={abnormal_scores.max():.6f}"
    )

    threshold = normal_scores.mean() + 3 * normal_scores.std()

    normal_fp = (normal_scores > threshold).sum()
    abnormal_detected = (abnormal_scores > threshold).sum()

    print(f"Threshold normal mean + 3std: {threshold:.6f}")
    print(f"Normal false positive: {normal_fp} / {len(normal_scores)}")
    print(f"Abnormal detected: {abnormal_detected} / {len(abnormal_scores)}")