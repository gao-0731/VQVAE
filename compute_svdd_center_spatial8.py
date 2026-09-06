import os
import numpy as np

import torch
import torch.nn as nn
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

model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage2_lightmask_v2_final.pth"

save_path = "/app/results/svdd_center_spatial8_stage2_v2.pth"


# =============================
# SVDD Head 8x8
# =============================

class SVDDHead8x8(nn.Module):
    """
    input:
        z_q [B,64,64,64]

    output:
        svdd_feat [B,128,8,8]
    """

    def __init__(self, in_dim=64, hidden_dim=128):
        super(SVDDHead8x8, self).__init__()

        self.net = nn.Sequential(
            # 64 -> 32
            nn.Conv2d(
                in_dim,
                hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(0.1, inplace=True),

            # 32 -> 16
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(0.1, inplace=True),

            # 16 -> 8
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, z_q):
        return self.net(z_q)


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

        # =============================
        # HU conversion
        # 必须和 Stage1 / Stage2_v2 一致
        # =============================

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))

        image = image * slope + intercept

        # =============================
        # Lung window
        # 必须和训练一致
        # =============================

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
# VQ-VAE Model
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

print("Loaded VQ-VAE model:", model_path)


# =============================
# SVDD Head
# =============================

svdd_head = SVDDHead8x8(
    in_dim=embedding_dim,
    hidden_dim=128
).to(device)

svdd_head.eval()

print("Created SVDDHead8x8")
print(svdd_head)


# =============================
# Extract z_q
# =============================

def extract_zq(model, x):
    """
    x: [B,1,256,256]

    return:
        z_q: [B,64,64,64]
        perplexity
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
# Compute spatial 8x8 center
# =============================

sum_feat = torch.zeros(
    1,
    128,
    8,
    8,
    device=device
)

num_samples = 0
perplexities = []

print("Computing spatial 8x8 SVDD center...")

with torch.no_grad():

    for step, batch in enumerate(loader):

        x, paths = batch
        x = x.to(device)

        z_q, perplexity = extract_zq(
            model,
            x
        )

        # z_q: [B,64,64,64]
        svdd_feat = svdd_head(
            z_q
        )

        # svdd_feat: [B,128,8,8]
        sum_feat += svdd_feat.sum(
            dim=0,
            keepdim=True
        )

        num_samples += svdd_feat.size(0)

        perplexities.append(perplexity.item())

        if step % 50 == 0:
            print(
                f"Step [{step}/{len(loader)}] | "
                f"num_samples: {num_samples} | "
                f"svdd_feat shape: {svdd_feat.shape} | "
                f"perplexity: {perplexity.item():.4f}"
            )


center_map = sum_feat / num_samples

print("\n==== Spatial 8x8 Center Computed ====")
print("Num samples:", num_samples)
print("Center map shape:", center_map.shape)
print("Center map mean:", center_map.mean().item())
print("Center map std:", center_map.std().item())
print("Average perplexity:", np.mean(perplexities))


# =============================
# Save center + head initial weights
# =============================

torch.save(
    {
        "center_map": center_map.detach().cpu(),

        # 这个很重要：
        # 训练 spatial SVDD 时必须加载同一个初始化的 head
        "svdd_head_state_dict": svdd_head.state_dict(),

        "num_samples": num_samples,
        "model_path": model_path,
        "normal_data_dir": normal_data_dir,

        "embedding_dim": embedding_dim,
        "hidden_dim": 128,
        "spatial_size": 8,

        "n_hiddens": n_hiddens,
        "n_residual_hiddens": n_residual_hiddens,
        "n_residual_layers": n_residual_layers,
        "n_embeddings": n_embeddings,
        "beta": beta,

        "type": "spatial8_svdd_center",
    },
    save_path
)

print("Saved spatial 8x8 SVDD center to:", save_path)
