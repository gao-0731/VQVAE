import os
import numpy as np
import torch
import torch.nn.functional as F

from torchvision import transforms
import torchvision.utils as vutils

import pydicom

from models.vqvae import VQVAE


# =============================
# Settings
# =============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

resize = 256

# 改成你想测试的 DICOM
dicom_path = "/app/data/temp/abnormal/54.dcm"

# 推荐先用 Stage2_v2 或 Stage4
model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage2_lightmask_v2_final.pth"
# model_path = "/app/results/vqvae_data_vqvae_ct_lung_region_masked_stage4_final.pth"

save_path = "/app/results/manual_mask_test.png"


# =============================
# VQ-VAE Parameters
# 必须和训练一致
# =============================

n_hiddens = 128
n_residual_hiddens = 64
n_residual_layers = 1

embedding_dim = 64
n_embeddings = 64
beta = 0.25


# =============================
# Manual mask area
# 这里自己改成病灶位置
# 坐标是 256x256 图像上的坐标
# =============================

y1, y2 = 170, 190
x1, x2 = 65, 100


# =============================
# Mask mode
# 可选: "black", "gray", "local_mean"
# =============================

# mask_mode = "black"
mask_mode = "gray"
# mask_mode = "local_mean"


# =============================
# Load DICOM
# =============================

def load_dicom_as_tensor(dicom_path):

    ds = pydicom.dcmread(dicom_path)

    image = ds.pixel_array.astype(np.float32)

    # HU conversion
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    image = image * slope + intercept

    # Lung window [-1000, 400]
    window_min = -1000.0
    window_max = 400.0

    image = np.clip(image, window_min, window_max)
    image = (image - window_min) / (window_max - window_min)

    image = torch.tensor(image).unsqueeze(0)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((resize, resize)),
        transforms.ToTensor()
    ])

    image = transform(image)

    # [1,256,256] -> [1,1,256,256]
    image = image.unsqueeze(0)

    return image


# =============================
# Load Model
# =============================

model = VQVAE(
    n_hiddens,
    n_residual_hiddens,
    n_residual_layers,
    n_embeddings,
    embedding_dim,
    beta
).to(device)

checkpoint = torch.load(
    model_path,
    map_location=device
)

if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

model.eval()

print("Loaded model:", model_path)


# =============================
# Forward no mask
# =============================

def forward_no_mask(model, x):

    z_e = model.encoder(x)
    z_e = model.pre_quantization_conv(z_e)

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

    (
        embedding_loss,
        z_q,
        perplexity,
        _,
        _,
        q_distance_map
    ) = model.vector_quantization(z_e)

    x_hat = model.decoder(z_q)

    return x_hat, q_distance_map, perplexity


# =============================
# Apply manual mask
# =============================

def apply_manual_mask(x, y1, y2, x1, x2, mode="black"):

    x_masked = x.clone()

    if mode == "black":
        fill_value = 0.0

    elif mode == "gray":
        fill_value = 0.08

    elif mode == "local_mean":
        patch = x[
            :,
            :,
            max(0, y1 - 16):min(256, y2 + 16),
            max(0, x1 - 16):min(256, x2 + 16)
        ]
        fill_value = patch.mean().item()

    else:
        raise ValueError(f"Unknown mask mode: {mode}")

    x_masked[:, :, y1:y2, x1:x2] = fill_value

    mask = torch.zeros_like(x)
    mask[:, :, y1:y2, x1:x2] = 1.0

    return x_masked, mask


# =============================
# Visualization
# =============================

def normalize_map(x):
    x = x.detach()
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


# =============================
# Run
# =============================

x = load_dicom_as_tensor(dicom_path).to(device)

x_masked, manual_mask = apply_manual_mask(
    x,
    y1,
    y2,
    x1,
    x2,
    mode=mask_mode
)

with torch.no_grad():

    # 原图普通重建
    recon_original, q_original, perp_original = forward_no_mask(
        model,
        x
    )

    # 手动 mask 后重建
    recon_masked, q_masked, perp_masked = forward_no_mask(
        model,
        x_masked
    )

    diff_original = torch.abs(x - recon_original)

    # 注意：这里是和原图 x 比，不是和 x_masked 比
    diff_masked = torch.abs(x - recon_masked)

    q_map = q_masked.unsqueeze(1)
    q_map = F.interpolate(
        q_map,
        size=x.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    grid = vutils.make_grid(
        torch.cat([
            x.cpu(),
            x_masked.cpu(),
            recon_original.cpu(),
            normalize_map(diff_original).cpu(),
            recon_masked.cpu(),
            normalize_map(diff_masked).cpu(),
            normalize_map(q_map).cpu(),
            manual_mask.cpu(),
        ], dim=0),
        nrow=8,
        normalize=True
    )

    vutils.save_image(
        grid,
        save_path
    )

print("Saved:", save_path)
print("Perplexity original:", perp_original.item())
print("Perplexity masked:", perp_masked.item())
print("Mask mode:", mask_mode)
print("Mask area:", y1, y2, x1, x2)