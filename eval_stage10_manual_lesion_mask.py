import os
import torch
import torch.nn.functional as F
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

# ------------------------------------------------------------
# 你要测试的异常 DICOM
# 如果想测试整个 abnormal 文件夹，保持 abnormal_dir 即可
# ------------------------------------------------------------
abnormal_dir = "/app/data/temp/abnormal"

# Stage1 ordinary VQ-VAE
stage1_model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage1_full_final.pth"

# Stage10 normal inpainting VQ-VAE
stage10_model_path = "/app/results/vqvae_ct_lungwindow_stage10_normal_inpainting_best.pth"

save_root = "/app/results/eval_stage10_manual_lesion_mask"
os.makedirs(save_root, exist_ok=True)

resize = 256

# ============================================================
# Manual lesion box
# ============================================================
# 你之前给的 512 坐标：
# left top     = (130, 340)
# right bottom = (200, 380)
#
# resize 到 256 后约等于：
# x = 65~100
# y = 170~190
#
# 注意这里格式是 y1, y2, x1, x2
# ============================================================

manual_center_y = 180
manual_center_x = 82

mask_size_list = [24, 32, 40, 48]

# 如果你想直接指定 box，也可以用这个：
# manual_box_256 = (170, 190, 65, 100)  # y1,y2,x1,x2
manual_box_256 = None

fill_mode = "local_mean"   # "local_mean" or "constant"
fill_value = 0.08
local_mean_pad = 8


# ============================================================
# VQ-VAE Parameters
# ============================================================

n_hiddens = 128
n_residual_hiddens = 64
n_residual_layers = 1

embedding_dim = 64
n_embeddings = 64
beta = 0.25


# ============================================================
# Utilities
# ============================================================

def collect_dicom_paths(root_dir):
    paths = []

    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".dcm"):
                paths.append(os.path.join(root, f))

    paths = sorted(paths)
    return paths


def load_dicom_lung_window(dicom_path):
    ds = pydicom.dcmread(dicom_path)

    image = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    image = image * slope + intercept

    # Lung window [-1000, 400]
    window_min = -1000.0
    window_max = 400.0

    image = np.clip(image, window_min, window_max)
    image = (image - window_min) / (window_max - window_min)

    image = Image.fromarray((image * 255).astype(np.uint8))
    image = image.resize((resize, resize), Image.BILINEAR)
    image = np.array(image).astype(np.float32) / 255.0

    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)

    return image


def normalize_map(x):
    x = x.detach()
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def get_box_from_center(cy, cx, size):
    half = size // 2

    y1 = max(0, cy - half)
    y2 = min(resize, cy + half)

    x1 = max(0, cx - half)
    x2 = min(resize, cx + half)

    # 如果 size 是奇数或靠边，补齐
    if y2 - y1 < size:
        if y1 == 0:
            y2 = min(resize, size)
        elif y2 == resize:
            y1 = max(0, resize - size)

    if x2 - x1 < size:
        if x1 == 0:
            x2 = min(resize, size)
        elif x2 == resize:
            x1 = max(0, resize - size)

    return y1, y2, x1, x2


def get_local_mean_fill(x, y1, y2, x1, x2, pad=8):
    """
    x: [1,1,H,W]
    return scalar
    """

    _, _, h, w = x.shape

    yy1 = max(0, y1 - pad)
    yy2 = min(h, y2 + pad)
    xx1 = max(0, x1 - pad)
    xx2 = min(w, x2 + pad)

    region = x[:, :, yy1:yy2, xx1:xx2].clone()

    inner_y1 = y1 - yy1
    inner_y2 = y2 - yy1
    inner_x1 = x1 - xx1
    inner_x2 = x2 - xx1

    valid = torch.ones_like(region)
    valid[:, :, inner_y1:inner_y2, inner_x1:inner_x2] = 0.0

    if valid.sum() < 1:
        return torch.tensor(fill_value, dtype=x.dtype, device=x.device)

    mean_value = (region * valid).sum() / (valid.sum() + 1e-8)

    return mean_value


def apply_manual_mask(x, y1, y2, x1, x2):
    """
    x: [1,1,256,256]
    """

    x_masked = x.clone()
    mask = torch.zeros_like(x)

    if fill_mode == "local_mean":
        fill = get_local_mean_fill(
            x,
            y1,
            y2,
            x1,
            x2,
            pad=local_mean_pad
        )
    elif fill_mode == "constant":
        fill = torch.tensor(
            fill_value,
            dtype=x.dtype,
            device=x.device
        )
    else:
        raise ValueError(f"Unknown fill_mode: {fill_mode}")

    x_masked[:, :, y1:y2, x1:x2] = fill
    mask[:, :, y1:y2, x1:x2] = 1.0

    return x_masked, mask


def make_overlay(x, mask):
    overlay = x.clone() * 0.75 + mask * 0.25
    overlay = torch.clamp(overlay, 0.0, 1.0)
    return overlay


# ============================================================
# Load model
# ============================================================

def load_vqvae_model(model_path):
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

    return model


def forward_vqvae(model, x):
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

    recon = model.decoder(z_q)

    return recon, z_q, q_distance_map, perplexity


stage1_model = load_vqvae_model(stage1_model_path)
stage10_model = load_vqvae_model(stage10_model_path)


# ============================================================
# Evaluation
# ============================================================

def compute_region_scores(x, recon, mask):
    diff = torch.abs(x - recon)

    whole_diff = diff.mean().item()

    region_diff = (
        diff * mask
    ).sum().item() / (mask.sum().item() + 1e-8)

    outside_mask = 1.0 - mask

    outside_diff = (
        diff * outside_mask
    ).sum().item() / (outside_mask.sum().item() + 1e-8)

    return diff, whole_diff, region_diff, outside_diff


def evaluate_one_dicom(dicom_path, case_idx):
    case_name = os.path.basename(dicom_path).replace(".dcm", "")

    case_save_dir = os.path.join(save_root, case_name)
    os.makedirs(case_save_dir, exist_ok=True)

    x = load_dicom_lung_window(dicom_path).to(device)

    records = []

    with torch.no_grad():
        stage1_recon, _, _, stage1_perplexity = forward_vqvae(stage1_model, x)
        stage1_diff = torch.abs(x - stage1_recon)

    print()
    print("=" * 80)
    print(f"Case {case_idx}: {case_name}")
    print("Stage1 ordinary diff:", stage1_diff.mean().item())

    grid_items = [
        x.cpu(),
        stage1_recon.cpu(),
        normalize_map(stage1_diff).cpu(),
    ]

    for mask_size in mask_size_list:
        if manual_box_256 is not None:
            y1, y2, x1, x2 = manual_box_256
        else:
            y1, y2, x1, x2 = get_box_from_center(
                manual_center_y,
                manual_center_x,
                mask_size
            )

        x_masked, mask = apply_manual_mask(
            x,
            y1,
            y2,
            x1,
            x2
        )

        with torch.no_grad():
            recon10, _, _, perp10 = forward_vqvae(stage10_model, x_masked)

        diff10, whole_diff, region_diff, outside_diff = compute_region_scores(
            x,
            recon10,
            mask
        )

        # Stage1 ordinary diff in same mask region
        stage1_region_diff = (
            stage1_diff * mask
        ).sum().item() / (mask.sum().item() + 1e-8)

        stage1_outside_diff = (
            stage1_diff * (1.0 - mask)
        ).sum().item() / ((1.0 - mask).sum().item() + 1e-8)

        region_gain_vs_stage1 = region_diff - stage1_region_diff
        outside_gain_vs_stage1 = outside_diff - stage1_outside_diff

        print(
            f"mask={mask_size}, "
            f"box y={y1}:{y2}, x={x1}:{x2}, "
            f"stage10 region_diff={region_diff:.6f}, "
            f"stage1 region_diff={stage1_region_diff:.6f}, "
            f"region_gain={region_gain_vs_stage1:.6f}, "
            f"outside_gain={outside_gain_vs_stage1:.6f}"
        )

        records.append({
            "case_name": case_name,
            "path": dicom_path,
            "mask_size": mask_size,

            "y1": y1,
            "y2": y2,
            "x1": x1,
            "x2": x2,

            "stage1_whole_diff": stage1_diff.mean().item(),
            "stage1_region_diff": stage1_region_diff,
            "stage1_outside_diff": stage1_outside_diff,

            "stage10_whole_diff": whole_diff,
            "stage10_region_diff": region_diff,
            "stage10_outside_diff": outside_diff,

            "region_gain_vs_stage1": region_gain_vs_stage1,
            "outside_gain_vs_stage1": outside_gain_vs_stage1,

            "stage1_perplexity": float(stage1_perplexity.item()),
            "stage10_perplexity": float(perp10.item()),
        })

        overlay = make_overlay(x, mask)

        grid_items.extend([
            overlay.cpu(),
            x_masked.cpu(),
            recon10.cpu(),
            normalize_map(diff10).cpu(),
        ])

    # save image
    # columns:
    # input | stage1 recon | stage1 diff |
    # mask overlay 24 | masked24 | recon24 | diff24 |
    # mask overlay 32 | masked32 | recon32 | diff32 | ...
    grid = vutils.make_grid(
        torch.cat(grid_items, dim=0),
        nrow=3 + 4 * len(mask_size_list),
        normalize=True
    )

    save_img_path = os.path.join(
        case_save_dir,
        f"{case_name}_stage10_manual_mask.png"
    )

    vutils.save_image(grid, save_img_path)

    print("Saved image:", save_img_path)

    return records


# ============================================================
# Run
# ============================================================

abnormal_paths = collect_dicom_paths(abnormal_dir)

print("Abnormal DICOM count:", len(abnormal_paths))

all_records = []

for idx, path in enumerate(abnormal_paths):
    recs = evaluate_one_dicom(path, idx)
    all_records.extend(recs)


# ============================================================
# Save CSV
# ============================================================

df = pd.DataFrame(all_records)

csv_path = os.path.join(
    save_root,
    "stage10_manual_lesion_mask_scores.csv"
)

df.to_csv(csv_path, index=False)

print()
print("=" * 80)
print("Saved CSV:", csv_path)
print("Saved images under:", save_root)

print()
print("Image columns:")
print("1 input")
print("2 stage1 ordinary recon")
print("3 stage1 ordinary diff")
for i, s in enumerate(mask_size_list):
    base = 4 + i * 4
    print(f"{base} mask{s} overlay")
    print(f"{base+1} mask{s} input")
    print(f"{base+2} stage10 recon mask{s}")
    print(f"{base+3} stage10 diff mask{s}")
print("=" * 80)