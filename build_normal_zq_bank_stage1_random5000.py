import os
import random
import torch
import torch.nn.functional as F
import pydicom
import numpy as np
from PIL import Image

from models.vqvae import VQVAE


# ============================================================
# Settings
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

normal_train_dir = "/app/data/train"

model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage1_full_final.pth"

save_path = "/app/results/normal_zq_bank_stage1_random5000.pt"

resize = 256

max_normal_samples = 5000
random_seed = 42


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

def collect_dicom_paths(root_dir, max_samples=None, random_sample=False, seed=42):
    paths = []

    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".dcm"):
                paths.append(os.path.join(root, f))

    paths = sorted(paths)

    print("Total DICOM files found:", len(paths))

    if max_samples is not None and len(paths) > max_samples:
        if random_sample:
            random.seed(seed)
            paths = random.sample(paths, max_samples)
            paths = sorted(paths)
        else:
            paths = paths[:max_samples]

    return paths


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
# Extract z_q
# ============================================================

def extract_zq(model, x):
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

    return z_q, perplexity


# ============================================================
# Build bank
# ============================================================

normal_paths = collect_dicom_paths(
    normal_train_dir,
    max_samples=max_normal_samples,
    random_sample=True,
    seed=random_seed
)

print("Selected prototype samples:", len(normal_paths))

bank = []

with torch.no_grad():
    for i, path in enumerate(normal_paths):
        try:
            x = load_dicom_as_tensor(path).to(device)

            z_q, perplexity = extract_zq(model, x)

            bank.append({
                "path": path,
                "case_name": os.path.basename(path).replace(".dcm", ""),
                "zq": z_q.detach().cpu(),
                "perplexity": float(perplexity.item()),
            })

            if (i + 1) % 50 == 0:
                print(f"Processed {i+1}/{len(normal_paths)}")

        except Exception as e:
            print("Skip error:", path)
            print(e)

save_data = {
    "bank": bank,
    "model_path": model_path,
    "normal_train_dir": normal_train_dir,
    "max_normal_samples": max_normal_samples,
    "random_seed": random_seed,
    "resize": resize,
    "n_samples_saved": len(bank),
}

torch.save(save_data, save_path)

print()
print("Saved normal z_q bank:")
print(save_path)
print("Bank size:", len(bank))