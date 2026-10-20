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

# 使用 Stage1 full VQ-VAE，避免 Stage2 masked reconstruction 黑点问题
model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage1_full_final.pth"

save_root = "/app/results/eval_stage5_v3_sliding_lung_gray_mask"

resize = 256
batch_size = 1
num_workers = 2

# sliding mask 参数
block_size = 40
stride = 16

# 肺内填充值
fill_value = 0.08

# block 至少多少比例在 lung mask 内，才参与 sliding
min_lung_ratio = 0.25

# 为了避免太慢，可限制保存图片数量；None 表示全部保存
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
# Utility functions
# ============================================================

def normalize_map(x):
    x = x.detach()
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def extract_recon_and_qdist(model, x):
    """
    x: [B,1,256,256]

    return:
        x_hat: [B,1,256,256]
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

    return x_hat, q_distance_map, perplexity


def make_lung_mask(x):
    """
    x: [1,1,256,256]

    return:
        lung_mask: [1,1,256,256]

    这个 mask 比较宽松，用于限制 sliding 只在肺区域附近。
    """

    lung = ((x > 0.02) & (x < 0.45)).float()

    # 轻微 dilation，连接肺区域
    lung = F.max_pool2d(
        lung,
        kernel_size=5,
        stride=1,
        padding=2
    )

    # 轻微 erosion，去掉最外层噪声
    lung = -F.max_pool2d(
        -lung,
        kernel_size=5,
        stride=1,
        padding=2
    )

    lung = (lung > 0.5).float()

    return lung


def apply_block_mask(x, y1, y2, x1, x2):
    """
    对指定 block 使用 lung-gray 填充
    """

    x_masked = x.clone()
    x_masked[:, :, y1:y2, x1:x2] = fill_value

    mask = torch.zeros_like(x)
    mask[:, :, y1:y2, x1:x2] = 1.0

    return x_masked, mask


def sliding_lung_gray_search(model, x):
    """
    对图像做 sliding lung-gray mask，寻找最可疑 block。

    x: [1,1,256,256]

    return:
        best_y1, best_y2, best_x1, best_x2
        best_score
        sliding_score_map: [1,1,256,256]
        best_mask: [1,1,256,256]
        best_masked_input: [1,1,256,256]
        best_recon: [1,1,256,256]
        best_diff: [1,1,256,256]
        lung_mask: [1,1,256,256]
    """

    lung_mask = make_lung_mask(x)

    sliding_score_map = torch.zeros_like(x)
    sliding_count_map = torch.zeros_like(x)

    best_score = -1.0
    best_coords = None
    best_mask = None
    best_masked_input = None
    best_recon = None
    best_diff = None

    # 用来判断 block 是否在肺区域内
    lung_block_ratio = F.avg_pool2d(
        lung_mask,
        kernel_size=block_size,
        stride=stride
    )

    block_rows = lung_block_ratio.shape[-2]
    block_cols = lung_block_ratio.shape[-1]

    total_blocks = block_rows * block_cols
    valid_blocks = 0

    with torch.no_grad():
        for by in range(block_rows):
            for bx in range(block_cols):

                lung_ratio = lung_block_ratio[0, 0, by, bx].item()

                if lung_ratio < min_lung_ratio:
                    continue

                valid_blocks += 1

                y1 = by * stride
                x1 = bx * stride
                y2 = y1 + block_size
                x2 = x1 + block_size

                # 防止越界
                if y2 > 256 or x2 > 256:
                    continue

                x_masked, mask = apply_block_mask(
                    x,
                    y1,
                    y2,
                    x1,
                    x2
                )

                recon_masked, qdist_masked, perp_masked = extract_recon_and_qdist(
                    model,
                    x_masked
                )

                diff = torch.abs(x - recon_masked)

                # 只计算当前 block 内的差异
                block_diff = diff[:, :, y1:y2, x1:x2]
                block_score = block_diff.mean().item()

                # 累积到 score map
                sliding_score_map[:, :, y1:y2, x1:x2] += block_score
                sliding_count_map[:, :, y1:y2, x1:x2] += 1.0

                if block_score > best_score:
                    best_score = block_score
                    best_coords = (y1, y2, x1, x2)
                    best_mask = mask.clone()
                    best_masked_input = x_masked.clone()
                    best_recon = recon_masked.clone()
                    best_diff = diff.clone()

    # 如果 lung mask 太严格导致没有 valid block，则退回全图 sliding
    if valid_blocks == 0:
        print("Warning: no valid lung blocks. Fallback to full-image sliding.")

        for y1 in range(0, 256 - block_size + 1, stride):
            for x1 in range(0, 256 - block_size + 1, stride):
                y2 = y1 + block_size
                x2 = x1 + block_size

                x_masked, mask = apply_block_mask(
                    x,
                    y1,
                    y2,
                    x1,
                    x2
                )

                recon_masked, qdist_masked, perp_masked = extract_recon_and_qdist(
                    model,
                    x_masked
                )

                diff = torch.abs(x - recon_masked)

                block_diff = diff[:, :, y1:y2, x1:x2]
                block_score = block_diff.mean().item()

                sliding_score_map[:, :, y1:y2, x1:x2] += block_score
                sliding_count_map[:, :, y1:y2, x1:x2] += 1.0

                if block_score > best_score:
                    best_score = block_score
                    best_coords = (y1, y2, x1, x2)
                    best_mask = mask.clone()
                    best_masked_input = x_masked.clone()
                    best_recon = recon_masked.clone()
                    best_diff = diff.clone()

    sliding_score_map = sliding_score_map / (sliding_count_map + 1e-8)

    y1, y2, x1, x2 = best_coords

    return (
        y1,
        y2,
        x1,
        x2,
        best_score,
        sliding_score_map,
        best_mask,
        best_masked_input,
        best_recon,
        best_diff,
        lung_mask,
        valid_blocks,
        total_blocks
    )


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
            recon_original, q_distance_map, perplexity = extract_recon_and_qdist(
                model,
                x
            )

            diff_original = torch.abs(x - recon_original)

            # --------------------------------------------------------
            # sliding lung-gray search
            # --------------------------------------------------------
            (
                y1,
                y2,
                x1,
                x2,
                best_score,
                sliding_score_map,
                best_mask,
                best_masked_input,
                best_recon,
                best_diff,
                lung_mask,
                valid_blocks,
                total_blocks
            ) = sliding_lung_gray_search(
                model,
                x
            )

            # --------------------------------------------------------
            # qdistance map upsample
            # --------------------------------------------------------
            q_map = q_distance_map.unsqueeze(1)
            q_map = F.interpolate(
                q_map,
                size=(256, 256),
                mode="bilinear",
                align_corners=False
            )

            # --------------------------------------------------------
            # Scores
            # --------------------------------------------------------
            ordinary_diff_score = diff_original.mean().item()
            targeted_diff_score = best_diff.mean().item()

            targeted_region_diff_score = (
                best_diff * best_mask
            ).sum().item() / (best_mask.sum().item() + 1e-8)

            sliding_mean = sliding_score_map.mean().item()
            sliding_max = sliding_score_map.max().item()
            sliding_top1 = topk_mean_score(sliding_score_map, k_ratio=0.01)

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
                        x.cpu(),                                      # 1 input
                        recon_original.cpu(),                         # 2 ordinary recon
                        normalize_map(diff_original).cpu(),           # 3 ordinary diff
                        lung_mask.cpu(),                              # 4 lung mask
                        normalize_map(sliding_score_map).cpu(),       # 5 sliding score map
                        best_mask.cpu(),                              # 6 best mask
                        best_masked_input.cpu(),                      # 7 masked input
                        best_recon.cpu(),                             # 8 targeted recon
                        normalize_map(best_diff).cpu(),               # 9 targeted diff
                        normalize_map(q_map).cpu(),                   # 10 qdistance original
                    ], dim=0),
                    nrow=10,
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

                "valid_blocks": valid_blocks,
                "total_blocks": total_blocks,

                "best_sliding_score": best_score,
                "sliding_mean": sliding_mean,
                "sliding_max": sliding_max,
                "sliding_top1percent": sliding_top1,

                "ordinary_diff": ordinary_diff_score,
                "targeted_diff": targeted_diff_score,
                "targeted_region_diff": targeted_region_diff_score,

                "qdist_mean": qdist_mean,
                "qdist_max": qdist_max,

                "perplexity_original": float(perplexity.item()),

                "image_path": image_save_path,
            })

            print(
                f"[{label}] {idx+1}/{len(dataset)} "
                f"{case_name} "
                f"mask=({y1}:{y2},{x1}:{x2}) "
                f"valid_blocks={valid_blocks}/{total_blocks} "
                f"best={best_score:.6f} "
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
    "stage5_v3_sliding_lung_gray_mask_scores.csv"
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
    "best_sliding_score",
    "sliding_mean",
    "sliding_max",
    "sliding_top1percent",
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