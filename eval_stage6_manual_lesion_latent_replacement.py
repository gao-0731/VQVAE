import os
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
import pydicom
import numpy as np
from PIL import Image

from models.vqvae import VQVAE


# ============================================================
# Settings
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 改成你要测试的 abnormal DICOM
dicom_path = "/app/data/temp/abnormal/54.dcm"

# Stage1 full VQ-VAE
model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage1_full_final.pth"

# Stage1 z_q center/std
stats_path = "/app/results/zq_stats_stage1_full.pth"

save_path = "/app/results/stage6_manual_lesion_latent_replacement.png"

resize = 256


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
# Manual lesion coordinates
# 你之前给的是 512x512 坐标：
# 左上角: x=130, y=340
# 右下角: x=200, y=380
#
# 转成 256x256：
# x=65~100
# y=170~190
#
# 转成 latent 64x64：
# x=16~25
# y=42~48
#
# 这里稍微扩大一点
# ============================================================

ly1, ly2 = 40, 50
lx1, lx2 = 14, 27

alpha_list = [0.3, 0.5, 0.7, 1.0]


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
# Functions
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


def soft_replace_latent_block(z_q, center_map, ly1, ly2, lx1, lx2, alpha):
    """
    z_corr = z_q outside lesion
           = (1-alpha)*z_q + alpha*center_map inside lesion
    """

    z_corr = z_q.clone()

    z_corr[:, :, ly1:ly2, lx1:lx2] = (
        (1.0 - alpha) * z_q[:, :, ly1:ly2, lx1:lx2]
        + alpha * center_map[:, :, ly1:ly2, lx1:lx2]
    )

    return z_corr


def make_image_mask_from_latent_box(x, ly1, ly2, lx1, lx2):
    """
    latent 64x64 -> image 256x256
    scale = 4
    """

    scale = 4

    y1 = ly1 * scale
    y2 = ly2 * scale
    x1 = lx1 * scale
    x2 = lx2 * scale

    mask = torch.zeros_like(x)
    mask[:, :, y1:y2, x1:x2] = 1.0

    return mask, y1, y2, x1, x2


# ============================================================
# Run
# ============================================================

x = load_dicom_as_tensor(dicom_path).to(device)

with torch.no_grad():

    # ordinary reconstruction
    recon_original, z_q, q_distance_map, perplexity = extract_zq_recon_qdist(
        model,
        x
    )

    diff_original = torch.abs(x - recon_original)

    # lesion image mask
    lesion_mask, y1_img, y2_img, x1_img, x2_img = make_image_mask_from_latent_box(
        x,
        ly1,
        ly2,
        lx1,
        lx2
    )

    print("Manual latent box:")
    print("latent y:", ly1, ly2, "latent x:", lx1, lx2)
    print("image y:", y1_img, y2_img, "image x:", x1_img, x2_img)

    # qdistance map
    q_map = q_distance_map.unsqueeze(1)
    q_map = F.interpolate(
        q_map,
        size=(256, 256),
        mode="bilinear",
        align_corners=False
    )

    images = [
        x.cpu(),
        recon_original.cpu(),
        normalize_map(diff_original).cpu(),
        lesion_mask.cpu(),
        normalize_map(q_map).cpu(),
    ]

    score_records = []

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

        diff_corr = torch.abs(x - recon_corr)

        region_diff = (
            diff_corr * lesion_mask
        ).sum().item() / (lesion_mask.sum().item() + 1e-8)

        whole_diff = diff_corr.mean().item()

        outside_mask = 1.0 - lesion_mask
        outside_diff = (
            diff_corr * outside_mask
        ).sum().item() / (outside_mask.sum().item() + 1e-8)

        score_records.append({
            "alpha": alpha,
            "whole_diff": whole_diff,
            "region_diff": region_diff,
            "outside_diff": outside_diff,
        })

        print(
            f"alpha={alpha:.1f} "
            f"whole_diff={whole_diff:.6f} "
            f"region_diff={region_diff:.6f} "
            f"outside_diff={outside_diff:.6f}"
        )

        images.extend([
            recon_corr.cpu(),
            normalize_map(diff_corr).cpu(),
        ])

    grid = vutils.make_grid(
        torch.cat(images, dim=0),
        nrow=5 + 2 * len(alpha_list),
        normalize=True
    )

    vutils.save_image(
        grid,
        save_path
    )

print("Saved:", save_path)