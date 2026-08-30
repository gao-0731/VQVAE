import os
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pydicom

from models.vqvae import VQVAE


# =============================
# Basic Settings
# =============================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

resize = 256
batch_size = 16


# =============================
# Model Parameters
# 必须和 Stage2_v2 一致
# =============================

n_hiddens = 128
n_residual_hiddens = 64
n_residual_layers = 1

embedding_dim = 64
n_embeddings = 64
beta = 0.25


# =============================
# Paths
# =============================

normal_data_dir = "/app/data/train"

# 推荐用 Stage2_v2，不用刚刚训练后的 global SVDD 模型
model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage2_lightmask_v2_final.pth"

save_path = "/app/results/zq_svdd_stats_stage2_v2.pth"


# =============================
# Dataset
# =============================

class DICOMDataset(Dataset):

    def __init__(self, data_path, transform=None):
        self.transform = transform
        self.dicom_files = []

        for root, _, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith(".dcm"):
                    self.dicom_files.append(
                        os.path.join(root, file)
                    )

        self.dicom_files = sorted(self.dicom_files)

        print(f"Loaded {len(self.dicom_files)} DICOM files from {data_path}")

    def __len__(self):
        return len(self.dicom_files)

    def __getitem__(self, idx):

        dicom_path = self.dicom_files[idx]

        ds = pydicom.dcmread(dicom_path)

        image = ds.pixel_array.astype(np.float32)

        # HU conversion
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))

        image = image * slope + intercept

        # Lung window
        window_min = -1000.0
        window_max = 400.0

        image = np.clip(image, window_min, window_max)
        image = (image - window_min) / (window_max - window_min)

        image = torch.tensor(image).unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return image, dicom_path


# =============================
# Transform
# =============================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((resize, resize)),
    transforms.ToTensor()
])


# =============================
# DataLoader
# =============================

dataset = DICOMDataset(
    normal_data_dir,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

print("Dataset size:", len(dataset))
print("Batches:", len(loader))


# =============================
# Model
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
# Extract z_q
# =============================

def extract_zq(model, x):
    """
    x: [B,1,256,256]
    return:
        z_q: [B,64,64,64]
    """

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

    return z_q, perplexity


# =============================
# First pass: sum
# =============================

sum_z = torch.zeros(
    1,
    embedding_dim,
    64,
    64,
    device=device
)

num_samples = 0
perplexities = []

print("First pass: computing mean...")

with torch.no_grad():

    for step, batch in enumerate(loader):

        x, paths = batch
        x = x.to(device)

        z_q, perplexity = extract_zq(model, x)

        sum_z += z_q.sum(dim=0, keepdim=True)
        num_samples += z_q.size(0)

        perplexities.append(perplexity.item())

        if step % 50 == 0:
            print(
                f"Step [{step}/{len(loader)}] | "
                f"num_samples: {num_samples} | "
                f"perplexity: {perplexity.item():.4f}"
            )


center_map = sum_z / num_samples

print("Center map computed.")
print("center_map shape:", center_map.shape)
print("center_map mean:", center_map.mean().item())
print("center_map std:", center_map.std().item())


# =============================
# Second pass: std
# =============================

sum_sq_diff = torch.zeros_like(center_map)

print("Second pass: computing std...")

with torch.no_grad():

    for step, batch in enumerate(loader):

        x, paths = batch
        x = x.to(device)

        z_q, perplexity = extract_zq(model, x)

        diff = z_q - center_map
        sum_sq_diff += (diff ** 2).sum(dim=0, keepdim=True)

        if step % 50 == 0:
            print(
                f"Step [{step}/{len(loader)}] | "
                f"computing std..."
            )


var_map = sum_sq_diff / num_samples
std_map = torch.sqrt(var_map + 1e-8)

print("Std map computed.")
print("std_map shape:", std_map.shape)
print("std_map mean:", std_map.mean().item())
print("std_map std:", std_map.std().item())
print("Average perplexity:", np.mean(perplexities))


# =============================
# Save
# =============================

torch.save(
    {
        "center_map": center_map.detach().cpu(),
        "std_map": std_map.detach().cpu(),
        "num_samples": num_samples,
        "model_path": model_path,
        "normal_data_dir": normal_data_dir,
        "embedding_dim": embedding_dim,
        "n_embeddings": n_embeddings,
        "beta": beta,
        "type": "position_aware_zq_svdd_stats",
    },
    save_path
)

print("Saved z_q SVDD stats to:", save_path)