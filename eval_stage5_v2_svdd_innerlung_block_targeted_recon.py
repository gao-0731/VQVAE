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


# ============================================================
# Settings
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

normal_data_dir = "/app/data/temp/normal"
abnormal_data_dir = "/app/data/temp/abnormal"

model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage1_full_final.pth"
stats_path = "/app/results/zq_stats_stage1_full.pth"

save_root = "/app/results/eval_stage5_v2_svdd_innerlung_block_targeted_recon"

resize = 256
batch_size = 1
num_workers = 2

# targeted mask size
block_size = 40

# lung-like gray value
fill_value = 0.08

# use lung constraint
use_lung_constraint = True

# block 至少多少比例在 lung mask 内才允许被选中
min_lung_ratio = 0.30

# fallback 时更宽松的比例
fallback_lung_ratio = 0.10

# 最多保存多少张图；None 表示全部保存
max_save_images = None


# ============================================================
# Model Parameters
# ============================================================

n_hiddens = 128
n_residual_hiddens = 64
n_residual_layers = 1

embedding_dim = 64
n_embeddings = 64
beta = 0.25


# ============================================================
# Dataset
# ============================================================

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

        # HU conversion
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        image = image * slope + intercept

        # lung window [-1000, 400]
        window_min = -1000.0
        window_max = 400.0

        image = np.clip(image, window_min, window_max)
        image = (image - window_min) / (window_max - window_min)

        # resize to 256 x 256
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = image.resize((resize, resize), Image.BILINEAR)
        image = np.array(image).astype(np.float32) / 255.0

        image = torch.tensor(image).unsqueeze(0)  # [1,256,256]

        case_name = os.path.basename(path).replace(".dcm", "")

        return image, path, case_name


# ============================================================
# Load model
# ============================================================

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


# ============================================================
# Load z_q stats
# ============================================================

stats = torch.load(stats_path, map_location=device)

center_map = stats["center_map"].to(device)
std_map = stats["std_map"].to(device)

print("Loaded stats:", stats_path)
print("center_map:", center_map.shape)
print("std_map:", std_map.shape)


# ============================================================
# Utility functions
# ============================================================

def normalize_map(x):
    x = x.detach()
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def extract_zq_and_recon(model, x):
    """
    x: [B,1,256,256]

    return:
        x_hat: [B,1,256,256]
        z_q: [B,64,64,64]
        q_distance_map: [B,64,64]
        perplexity
    """

    z_e = model.encoder(x)
    z_e = model.pre_quantization_conv(z_e)

    if hasattr(model, "pos_bias"):
        if (
            model.pos_bias.shape[2] == z_e.shape[2]
            and model.pos_bias.shape[3] == z_e.shape[3]
        ):
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


def make_lung_mask(x):
    """
    x: [1,1,256,256]

    return:
        lung_mask: [1,1,256,256]

    这个版本比之前宽松：
    - 使用 x > 0.02 and x < 0.45
    - 轻微 dilation 连接肺区域
    - 轻微 erosion 去掉最外层噪声
    - 不会把肺底和胸膜附近删得太厉害
    """

    lung = ((x > 0.02) & (x < 0.45)).float()

    # slight dilation
    lung = F.max_pool2d(
        lung,
        kernel_size=5,
        stride=1,
        padding=2
    )

    # slight erosion
    lung = -F.max_pool2d(
        -lung,
        kernel_size=5,
        stride=1,
        padding=2
    )

    lung = (lung > 0.5).float()

    return lung


def get_svdd_map(z_q, center_map, std_map):
    """
    z_q: [B,64,64,64]
    center_map/std_map: [1,64,64,64]

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


def find_best_mask_from_svdd_block_average(svdd_map_256, x):
    """
    svdd_map_256: [1,1,256,256]
    x: [1,1,256,256]

    return:
        y1, y2, x1, x2,
        best_score,
        lung_mask,
        block_score_map_256,
        used_fallback
    """

    if use_lung_constraint:
        lung_mask = make_lung_mask(x)
        score = svdd_map_256 * lung_mask
    else:
        lung_mask = torch.ones_like(svdd_map_256)
        score = svdd_map_256.clone()

    # ------------------------------------------------------------
    # block average score
    # block_score shape:
    # [1,1,256-block_size+1,256-block_size+1]
    # ------------------------------------------------------------
    block_score = F.avg_pool2d(
        score,
        kernel_size=block_size,
        stride=1
    )

    used_fallback = False

    if use_lung_constraint:
        lung_block_ratio = F.avg_pool2d(
            lung_mask,
            kernel_size=block_size,
            stride=1
        )

        valid = lung_block_ratio >= min_lung_ratio

        if valid.sum().item() > 0:
            block_score = block_score.masked_fill(~valid, -1e9)

        else:
            used_fallback = True
            print(
                "Warning: no valid block with "
                f"min_lung_ratio={min_lung_ratio}. "
                f"Fallback to ratio={fallback_lung_ratio}."
            )

            valid_fallback = lung_block_ratio >= fallback_lung_ratio

            if valid_fallback.sum().item() > 0:
                block_score = F.avg_pool2d(
                    score,
                    kernel_size=block_size,
                    stride=1
                )
                block_score = block_score.masked_fill(~valid_fallback, -1e9)

            else:
                print(
                    "Warning: still no valid lung block. "
                    "Using lung-constrained score without ratio filtering."
                )

                block_score = F.avg_pool2d(
                    score,
                    kernel_size=block_size,
                    stride=1
                )

    # ------------------------------------------------------------
    # Find max block
    # ------------------------------------------------------------
    flat_idx = torch.argmax(block_score.view(-1))

    h = block_score.shape[-2]
    w = block_score.shape[-1]

    by = int(flat_idx // w)
    bx = int(flat_idx % w)

    y1 = by
    y2 = by + block_size
    x1 = bx
    x2 = bx + block_size

    best_score = block_score[0, 0, by, bx].item()

    # ------------------------------------------------------------
    # Visualization map
    # Replace invalid -1e9 values with min valid value
    # ------------------------------------------------------------
    block_score_vis = block_score.clone()
    valid_values = block_score_vis[block_score_vis > -1e8]

    if valid_values.numel() > 0:
        min_valid = valid_values.min()
        block_score_vis[block_score_vis < -1e8] = min_valid
    else:
        block_score_vis[block_score_vis < -1e8] = 0.0

    block_score_map_256 = F.interpolate(
        block_score_vis,
        size=(256, 256),
        mode="bilinear",
        align_corners=False
    )

    return (
        y1,
        y2,
        x1,
        x2,
        best_score,
        lung_mask,
        block_score_map_256,
        used_fallback
    )


def apply_targeted_mask(x, y1, y2, x1, x2):
    """
    用 lung-like gray value 填充 targeted region
    """

    x_masked = x.clone()
    x_masked[:, :, y1:y2, x1:x2] = fill_value

    mask = torch.zeros_like(x)
    mask[:, :, y1:y2, x1:x2] = 1.0

    return x_masked, mask


def topk_mean_score(score_map, k_ratio=0.01):
    """
    score_map: [1,1,H,W]
    计算 top-k mean，用于 case-level score
    """

    flat = score_map.view(-1)
    k = max(1, int(flat.numel() * k_ratio))
    vals, _ = torch.topk(flat, k)
    return vals.mean().item()


# ============================================================
# Evaluate dataset
# ============================================================

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

            # --------------------------------------------------------
            # ordinary reconstruction
            # --------------------------------------------------------
            recon_original, z_q, q_distance_map, perplexity = extract_zq_and_recon(model, x)
            diff_original = torch.abs(x - recon_original)

            # --------------------------------------------------------
            # SVDD map
            # --------------------------------------------------------
            svdd_map_64, svdd_map_256 = get_svdd_map(
                z_q,
                center_map,
                std_map
            )

            # --------------------------------------------------------
            # Find best targeted region
            # --------------------------------------------------------
            (
                y1,
                y2,
                x1,
                x2,
                best_score,
                lung_mask,
                block_score_map_256,
                used_fallback
            ) = find_best_mask_from_svdd_block_average(
                svdd_map_256,
                x
            )

            # --------------------------------------------------------
            # targeted mask
            # --------------------------------------------------------
            x_masked, targeted_mask = apply_targeted_mask(
                x,
                y1,
                y2,
                x1,
                x2
            )

            # --------------------------------------------------------
            # targeted reconstruction
            # --------------------------------------------------------
            recon_targeted, z_q_targeted, q_distance_map_targeted, perplexity_targeted = extract_zq_and_recon(
                model,
                x_masked
            )

            diff_targeted = torch.abs(x - recon_targeted)

            # --------------------------------------------------------
            # qdistance maps
            # --------------------------------------------------------
            q_map = q_distance_map.unsqueeze(1)
            q_map = F.interpolate(
                q_map,
                size=(256, 256),
                mode="bilinear",
                align_corners=False
            )

            q_map_targeted = q_distance_map_targeted.unsqueeze(1)
            q_map_targeted = F.interpolate(
                q_map_targeted,
                size=(256, 256),
                mode="bilinear",
                align_corners=False
            )

            # --------------------------------------------------------
            # Scores
            # --------------------------------------------------------
            ordinary_diff_score = diff_original.mean().item()
            targeted_diff_score = diff_targeted.mean().item()

            targeted_region_diff_score = (
                diff_targeted * targeted_mask
            ).sum().item() / (targeted_mask.sum().item() + 1e-8)

            svdd_mean = svdd_map_256.mean().item()
            svdd_max = svdd_map_256.max().item()
            svdd_top1 = topk_mean_score(svdd_map_256, k_ratio=0.01)

            block_score_mean = block_score_map_256.mean().item()
            block_score_max = block_score_map_256.max().item()

            qdist_mean = q_map.mean().item()
            qdist_max = q_map.max().item()

            # --------------------------------------------------------
            # Save image
            # --------------------------------------------------------
            should_save = True
            if max_save_images is not None and idx >= max_save_images:
                should_save = False

            image_save_path = ""

            if should_save:
                grid = vutils.make_grid(
                    torch.cat([
                        x.cpu(),                                  # 1 input
                        recon_original.cpu(),                     # 2 ordinary recon
                        normalize_map(diff_original).cpu(),       # 3 ordinary diff
                        normalize_map(svdd_map_256).cpu(),        # 4 svdd map
                        normalize_map(block_score_map_256).cpu(), # 5 block average svdd map
                        lung_mask.cpu(),                          # 6 lung mask
                        targeted_mask.cpu(),                      # 7 selected targeted mask
                        x_masked.cpu(),                           # 8 masked input
                        recon_targeted.cpu(),                     # 9 targeted recon
                        normalize_map(diff_targeted).cpu(),       # 10 targeted diff
                        normalize_map(q_map).cpu(),               # 11 qdistance original
                        normalize_map(q_map_targeted).cpu(),      # 12 qdistance targeted
                    ], dim=0),
                    nrow=12,
                    normalize=True
                )

                image_save_path = os.path.join(
                    save_dir,
                    f"{idx:04d}_{case_name}.png"
                )

                vutils.save_image(grid, image_save_path)

            # --------------------------------------------------------
            # Record
            # --------------------------------------------------------
            records.append({
                "label": label,
                "case_name": case_name,
                "path": path,

                "y1": y1,
                "y2": y2,
                "x1": x1,
                "x2": x2,

                "used_fallback": used_fallback,

                "svdd_best_block_score": best_score,
                "svdd_mean": svdd_mean,
                "svdd_max": svdd_max,
                "svdd_top1percent": svdd_top1,

                "block_score_mean": block_score_mean,
                "block_score_max": block_score_max,

                "ordinary_diff": ordinary_diff_score,
                "targeted_diff": targeted_diff_score,
                "targeted_region_diff": targeted_region_diff_score,

                "qdist_mean": qdist_mean,
                "qdist_max": qdist_max,

                "perplexity_original": float(perplexity.item()),
                "perplexity_targeted": float(perplexity_targeted.item()),

                "image_path": image_save_path,
            })

            fallback_text = "fallback" if used_fallback else "normal"

            print(
                f"[{label}] {idx+1}/{len(dataset)} "
                f"{case_name} "
                f"mask=({y1}:{y2},{x1}:{x2}) "
                f"{fallback_text} "
                f"best_block={best_score:.6f} "
                f"ordinary={ordinary_diff_score:.6f} "
                f"targeted={targeted_diff_score:.6f} "
                f"region={targeted_region_diff_score:.6f}"
            )

    return records


# ============================================================
# Run
# ============================================================

os.makedirs(save_root, exist_ok=True)

all_records = []

all_records += evaluate_dataset(normal_data_dir, "normal")
all_records += evaluate_dataset(abnormal_data_dir, "abnormal")

df = pd.DataFrame(all_records)

csv_path = os.path.join(
    save_root,
    "stage5_v2_svdd_innerlung_block_targeted_recon_scores.csv"
)

df.to_csv(csv_path, index=False)

print()
print("Saved CSV:", csv_path)


# ============================================================
# Summary
# ============================================================

score_names = [
    "ordinary_diff",
    "targeted_diff",
    "targeted_region_diff",
    "svdd_best_block_score",
    "svdd_mean",
    "svdd_max",
    "svdd_top1percent",
    "qdist_mean",
    "qdist_max",
]

for score_name in score_names:
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


# ============================================================
# Fallback summary
# ============================================================

print()
print("[fallback summary]")

for label in ["normal", "abnormal"]:
    sub = df[df["label"] == label]
    fallback_count = sub["used_fallback"].sum()
    total = len(sub)

    print(f"{label}: fallback {fallback_count} / {total}")