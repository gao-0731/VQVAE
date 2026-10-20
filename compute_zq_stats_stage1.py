import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pydicom
import numpy as np
from PIL import Image

from models.vqvae import VQVAE


# =============================
# Settings
# =============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_data_dir = "/app/data/train"

model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage1_full_final.pth"

save_path = "/app/results/zq_stats_stage1_full.pth"

resize = 256
batch_size = 16
num_workers = 4


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
    def __init__(self, root_dir):
        self.paths = []

        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(".dcm"):
                    self.paths.append(os.path.join(root, f))

        self.paths = sorted(self.paths)
        print(f"Found {len(self.paths)} DICOM files in {root_dir}")

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

        return image, path


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
# Extract z_q
# =============================

def extract_zq(model, x):
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

    return z_q


# =============================
# Compute center and std map
# =============================

dataset = DicomDataset(train_data_dir)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

sum_zq = None
sum_zq2 = None
count = 0

with torch.no_grad():
    for i, (x, paths) in enumerate(loader):
        x = x.to(device)

        z_q = extract_zq(model, x)

        # z_q: [B, C, H, W]
        if sum_zq is None:
            sum_zq = torch.zeros_like(z_q[:1])
            sum_zq2 = torch.zeros_like(z_q[:1])

        sum_zq += z_q.sum(dim=0, keepdim=True)
        sum_zq2 += (z_q ** 2).sum(dim=0, keepdim=True)
        count += z_q.shape[0]

        if i % 20 == 0:
            print(f"Processed {count} images")

center_map = sum_zq / count
var_map = sum_zq2 / count - center_map ** 2
std_map = torch.sqrt(torch.clamp(var_map, min=1e-8))

torch.save(
    {
        "center_map": center_map.cpu(),
        "std_map": std_map.cpu(),
        "count": count,
        "model_path": model_path
    },
    save_path
)

print("Saved:", save_path)
print("center_map:", center_map.shape)
print("std_map:", std_map.shape)
print("count:", count)