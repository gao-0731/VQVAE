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
# 改成你要测试的 abnormal DICOM
# ------------------------------------------------------------
dicom_path = "/app/data/temp/abnormal/54.dcm"

# Stage1 full VQ-VAE
model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage1_full_final.pth"

# Stage1 z_q center/std
stats_path = "/app/results/zq_stats_stage1_full.pth"

save_root = "/app/results/eval_stage6_manual_summary"
os.makedirs(save_root, exist_ok=True)

resize = 256

# soft replacement strengths
alpha_list = [0.3, 0.5, 0.7, 1.0]

# ============================================================
# Manual latent boxes
#
# 你之前 lesion 的 512 坐标：
# x=130~200, y=340~380
#
# 256 坐标：
# x=65~100, y=170~190
#
# latent 64 坐标：
# x=16~25, y=42~48
#
# box_original: 原来的手动框，稍微扩大
# box_large: 更大区域，用来确认是不是 latent 覆盖不够
# ============================================================

manual_boxes = [
    {
        "name": "box_original",
        "ly1": 40,
        "ly2": 50,
        "lx1": 14,
        "lx2": 27,
    },
    {
        "name": "box_large",
        "ly1": 36,
        "ly2": 54,
        "lx1": 10,
        "lx2": 31,
    },
]


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
# Load DICOM
# ============================================================

def load_dicom_as_tensor(dicom_path):
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


# ============================================================
# Load Model
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


def extract_zq_recon_qdist(model, x):
    """
    x: [1,1,256,256]

    return:
        recon: [1,1,256,256]
        z_q: [1,64,64,64]
        q_distance_map: [1,64,64]
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

    recon = model.decoder(z_q)

    return recon, z_q, q_distance_map, perplexity


def latent_to_image_box(ly1, ly2, lx1, lx2):
    scale = 4

    y1 = ly1 * scale
    y2 = ly2 * scale
    x1 = lx1 * scale
    x2 = lx2 * scale

    return y1, y2, x1, x2


def make_image_mask_from_latent_box(x, ly1, ly2, lx1, lx2):
    y1, y2, x1, x2 = latent_to_image_box(ly1, ly2, lx1, lx2)

    mask = torch.zeros_like(x)
    mask[:, :, y1:y2, x1:x2] = 1.0

    return mask, y1, y2, x1, x2


def make_overlay(x, mask):
    """
    灰度 overlay，用来确认 mask 是否盖住病灶。
    白色区域会叠加到原图上。
    """
    overlay = x.clone() * 0.75 + mask * 0.25
    overlay = torch.clamp(overlay, 0.0, 1.0)
    return overlay


def soft_replace_latent_block(z_q, center_map, ly1, ly2, lx1, lx2, alpha):
    """
    z_corr = z_q outside selected block
           = (1-alpha)*z_q + alpha*center_map inside selected block
    """

    z_corr = z_q.clone()

    z_corr[:, :, ly1:ly2, lx1:lx2] = (
        (1.0 - alpha) * z_q[:, :, ly1:ly2, lx1:lx2]
        + alpha * center_map[:, :, ly1:ly2, lx1:lx2]
    )

    return z_corr


def compute_scores(x, recon, region_mask):
    diff = torch.abs(x - recon)

    whole_diff = diff.mean().item()

    region_diff = (
        diff * region_mask
    ).sum().item() / (region_mask.sum().item() + 1e-8)

    outside_mask = 1.0 - region_mask
    outside_diff = (
        diff * outside_mask
    ).sum().item() / (outside_mask.sum().item() + 1e-8)

    return diff, whole_diff, region_diff, outside_diff


# ============================================================
# Run
# ============================================================

case_name = os.path.basename(dicom_path).replace(".dcm", "")

x = load_dicom_as_tensor(dicom_path).to(device)

records = []

with torch.no_grad():

    # --------------------------------------------------------
    # Ordinary reconstruction
    # --------------------------------------------------------
    recon_original, z_q, q_distance_map, perplexity = extract_zq_recon_qdist(
        model,
        x
    )

    diff_original = torch.abs(x - recon_original)

    # qdistance map
    q_map = q_distance_map.unsqueeze(1)
    q_map = F.interpolate(
        q_map,
        size=(256, 256),
        mode="bilinear",
        align_corners=False
    )

    ordinary_whole_diff = diff_original.mean().item()

    print()
    print("Case:", case_name)
    print("DICOM:", dicom_path)
    print("ordinary_diff:", ordinary_whole_diff)
    print("perplexity:", float(perplexity.item()))

    # --------------------------------------------------------
    # Evaluate each manual box
    # --------------------------------------------------------
    for box in manual_boxes:

        box_name = box["name"]

        ly1 = box["ly1"]
        ly2 = box["ly2"]
        lx1 = box["lx1"]
        lx2 = box["lx2"]

        lesion_mask, y1_img, y2_img, x1_img, x2_img = make_image_mask_from_latent_box(
            x,
            ly1,
            ly2,
            lx1,
            lx2
        )

        overlay = make_overlay(x, lesion_mask)

        print()
        print("=" * 60)
        print("Box:", box_name)
        print("latent y:", ly1, ly2, "latent x:", lx1, lx2)
        print("image  y:", y1_img, y2_img, "image  x:", x1_img, x2_img)

        # ordinary region diff under this mask
        ordinary_region_diff = (
            diff_original * lesion_mask
        ).sum().item() / (lesion_mask.sum().item() + 1e-8)

        ordinary_outside_diff = (
            diff_original * (1.0 - lesion_mask)
        ).sum().item() / ((1.0 - lesion_mask).sum().item() + 1e-8)

        print(
            f"ordinary: "
            f"whole={ordinary_whole_diff:.6f}, "
            f"region={ordinary_region_diff:.6f}, "
            f"outside={ordinary_outside_diff:.6f}"
        )

        # image list for this box
        images = [
            x.cpu(),                                  # 1 input
            recon_original.cpu(),                     # 2 ordinary recon
            normalize_map(diff_original).cpu(),       # 3 ordinary diff
            overlay.cpu(),                            # 4 input + mask overlay
            lesion_mask.cpu(),                        # 5 binary mask
            normalize_map(q_map).cpu(),               # 6 qdistance
        ]

        # save ordinary record
        records.append({
            "case_name": case_name,
            "dicom_path": dicom_path,
            "box_name": box_name,

            "latent_y1": ly1,
            "latent_y2": ly2,
            "latent_x1": lx1,
            "latent_x2": lx2,

            "image_y1": y1_img,
            "image_y2": y2_img,
            "image_x1": x1_img,
            "image_x2": x2_img,

            "alpha": "ordinary",
            "whole_diff": ordinary_whole_diff,
            "region_diff": ordinary_region_diff,
            "outside_diff": ordinary_outside_diff,
            "perplexity": float(perplexity.item()),
        })

        # ----------------------------------------------------
        # Multi-alpha soft replacement
        # ----------------------------------------------------
        for alpha in alpha_list:

            z_corr = soft_replace_latent_block(
                z_q,
                center_map,
                ly1,
                ly2,
                lx1,
                lx2,
                alpha
            )

            recon_corr = model.decoder(z_corr)

            diff_corr, whole_diff, region_diff, outside_diff = compute_scores(
                x,
                recon_corr,
                lesion_mask
            )

            print(
                f"alpha={alpha:.1f}: "
                f"whole={whole_diff:.6f}, "
                f"region={region_diff:.6f}, "
                f"outside={outside_diff:.6f}, "
                f"region_gain={region_diff - ordinary_region_diff:.6f}"
            )

            images.extend([
                recon_corr.cpu(),
                normalize_map(diff_corr).cpu(),
            ])

            records.append({
                "case_name": case_name,
                "dicom_path": dicom_path,
                "box_name": box_name,

                "latent_y1": ly1,
                "latent_y2": ly2,
                "latent_x1": lx1,
                "latent_x2": lx2,

                "image_y1": y1_img,
                "image_y2": y2_img,
                "image_x1": x1_img,
                "image_x2": x2_img,

                "alpha": alpha,
                "whole_diff": whole_diff,
                "region_diff": region_diff,
                "outside_diff": outside_diff,
                "region_gain_vs_ordinary": region_diff - ordinary_region_diff,
                "outside_gain_vs_ordinary": outside_diff - ordinary_outside_diff,
                "perplexity": float(perplexity.item()),
            })

        # ----------------------------------------------------
        # Save grid for this box
        # ----------------------------------------------------
        grid = vutils.make_grid(
            torch.cat(images, dim=0),
            nrow=6 + 2 * len(alpha_list),
            normalize=True
        )

        save_img_path = os.path.join(
            save_root,
            f"{case_name}_{box_name}_manual_latent_replacement.png"
        )

        vutils.save_image(grid, save_img_path)

        print("Saved image:", save_img_path)

# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------

df = pd.DataFrame(records)

csv_path = os.path.join(
    save_root,
    f"{case_name}_stage6_manual_summary.csv"
)

df.to_csv(csv_path, index=False)

print()
print("Saved CSV:", csv_path)

print()
print("Image columns for each box:")
print("1 input")
print("2 ordinary recon")
print("3 ordinary diff")
print("4 input + mask overlay")
print("5 binary mask")
print("6 qdistance")
print("7 alpha=0.3 corrected recon")
print("8 alpha=0.3 corrected diff")
print("9 alpha=0.5 corrected recon")
print("10 alpha=0.5 corrected diff")
print("11 alpha=0.7 corrected recon")
print("12 alpha=0.7 corrected diff")
print("13 alpha=1.0 corrected recon")
print("14 alpha=1.0 corrected diff")