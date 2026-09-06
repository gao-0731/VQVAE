import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

import pydicom

from models.vqvae import VQVAE


# =============================
# Basic Settings
# =============================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =============================
# Training Parameters
# =============================

batch_size = 16
n_epochs = 100
learning_rate = 5e-5

resize = 256

filename = "vqvae_ct_lungwindow_stage3_spatial8_svdd"


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
# SVDD Parameters
# =============================

lambda_svdd = 0.001
svdd_hidden_dim = 128


# =============================
# Paths
# =============================

train_data_dir = "/app/data/train"
val_data_dir = "/app/data/validation"

result_dir = "/app/results"
os.makedirs(result_dir, exist_ok=True)

pretrained_model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage2_lightmask_v2_final.pth"

spatial_center_path = "/app/results/svdd_center_spatial8_stage2_v2.pth"

run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

writer = SummaryWriter(
    log_dir=f"./runs/{filename}_{run_name}"
)

print("Run name:", run_name)
print("Filename:", filename)
print("Pretrained model:", pretrained_model_path)
print("Spatial center path:", spatial_center_path)
print("TensorBoard logdir:", f"./runs/{filename}_{run_name}")


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

        return image


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

train_dataset = DICOMDataset(
    train_data_dir,
    transform=transform
)

val_dataset = DICOMDataset(
    val_data_dir,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))


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
    pretrained_model_path,
    map_location=device
)

if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

print("Loaded Stage2_v2 VQ-VAE:", pretrained_model_path)


# =============================
# Load Spatial Center + Head Init
# =============================

center_ckpt = torch.load(
    spatial_center_path,
    map_location=device
)

center_map = center_ckpt["center_map"].to(device)

print("Loaded spatial center:", spatial_center_path)
print("Center map shape:", center_map.shape)
print("Center map mean:", center_map.mean().item())
print("Center map std:", center_map.std().item())


svdd_head = SVDDHead8x8(
    in_dim=embedding_dim,
    hidden_dim=svdd_hidden_dim
).to(device)

svdd_head.load_state_dict(
    center_ckpt["svdd_head_state_dict"]
)

print("Loaded SVDDHead8x8 initial state.")


# =============================
# Optimizer
# =============================

# 第一版：VQ-VAE 和 SVDDHead 都训练
# 如果重建变差，后面可以改成只训练 svdd_head
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(svdd_head.parameters()),
    lr=learning_rate
)


# =============================
# Helper: Save Checkpoint
# =============================

def save_checkpoint(model, svdd_head, optimizer, epoch, loss, name):

    save_path = os.path.join(
        result_dir,
        f"vqvae_data_{name}.pth"
    )

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "svdd_head": svdd_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss,

            "n_hiddens": n_hiddens,
            "n_residual_hiddens": n_residual_hiddens,
            "n_residual_layers": n_residual_layers,

            "embedding_dim": embedding_dim,
            "n_embeddings": n_embeddings,
            "beta": beta,

            "svdd_hidden_dim": svdd_hidden_dim,
            "lambda_svdd": lambda_svdd,

            "spatial_center_path": spatial_center_path,
            "pretrained_model_path": pretrained_model_path,
            "filename": filename,
        },
        save_path
    )

    print("Saved model:", save_path)


# =============================
# Forward with z_q
# =============================

def forward_with_zq(model, x):
    """
    x: [B,1,256,256]

    returns:
        embedding_loss
        x_hat
        perplexity
        q_distance_map
        z_q [B,64,64,64]
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

    x_hat = model.decoder(z_q)

    return (
        embedding_loss,
        x_hat,
        perplexity,
        q_distance_map,
        z_q
    )


# =============================
# Compute Losses
# =============================

def compute_losses(model, svdd_head, x, center_map):

    (
        embedding_loss,
        x_hat,
        perplexity,
        q_distance_map,
        z_q
    ) = forward_with_zq(
        model,
        x
    )

    # Reconstruction loss
    full_loss = torch.mean(
        torch.abs(x_hat - x)
    )

    # Spatial SVDD
    svdd_feat = svdd_head(
        z_q
    )

    # svdd_feat:  [B,128,8,8]
    # center_map: [1,128,8,8]
    svdd_map = torch.mean(
        (svdd_feat - center_map) ** 2,
        dim=1,
        keepdim=True
    )

    svdd_loss = torch.mean(
        svdd_map
    )

    loss = (
        full_loss
        + embedding_loss
        + lambda_svdd * svdd_loss
    )

    return (
        loss,
        full_loss,
        embedding_loss,
        svdd_loss,
        perplexity,
        x_hat,
        q_distance_map,
        z_q,
        svdd_map
    )


# =============================
# Visualization Helper
# =============================

def normalize_map(x):
    x = x.detach()
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + 1e-8)


def upsample_map(score_map, target_size=(256, 256)):
    return F.interpolate(
        score_map,
        size=target_size,
        mode="bilinear",
        align_corners=False
    )


# =============================
# Training
# =============================

best_val_loss = float("inf")
global_step = 0

print("Start Stage3 Spatial8 SVDD training...")

for epoch in range(n_epochs):

    # =============================
    # Train
    # =============================

    model.train()
    svdd_head.train()

    train_losses = []
    train_full_losses = []
    train_embedding_losses = []
    train_svdd_losses = []
    train_perplexities = []

    for batch_idx, x in enumerate(train_loader):

        x = x.to(device)

        optimizer.zero_grad()

        (
            loss,
            full_loss,
            embedding_loss,
            svdd_loss,
            perplexity,
            x_hat,
            q_distance_map,
            z_q,
            svdd_map
        ) = compute_losses(
            model,
            svdd_head,
            x,
            center_map
        )

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        train_full_losses.append(full_loss.item())
        train_embedding_losses.append(embedding_loss.item())
        train_svdd_losses.append(svdd_loss.item())
        train_perplexities.append(perplexity.item())

        writer.add_scalar(
            "Train/Loss",
            loss.item(),
            global_step
        )

        writer.add_scalar(
            "Train/Full_Loss",
            full_loss.item(),
            global_step
        )

        writer.add_scalar(
            "Train/Embedding_Loss",
            embedding_loss.item(),
            global_step
        )

        writer.add_scalar(
            "Train/SVDD_Loss",
            svdd_loss.item(),
            global_step
        )

        writer.add_scalar(
            "Train/Lambda_SVDD_x_Loss",
            lambda_svdd * svdd_loss.item(),
            global_step
        )

        writer.add_scalar(
            "Train/Perplexity",
            perplexity.item(),
            global_step
        )

        if global_step % 200 == 0:

            n_show = min(8, x.size(0))

            full_diff = torch.abs(x - x_hat)

            svdd_map_up = upsample_map(
                svdd_map,
                x.shape[-2:]
            )

            svdd_map_vis = normalize_map(
                svdd_map_up
            )

            full_diff_vis = normalize_map(
                full_diff
            )

            grid = vutils.make_grid(
                torch.cat([
                    x[:n_show],
                    x_hat[:n_show],
                    full_diff_vis[:n_show],
                    svdd_map_vis[:n_show]
                ], dim=0),
                nrow=n_show,
                normalize=True
            )

            writer.add_image(
                "Train/Input_Recon_FullDiff_SpatialSVDD",
                grid,
                global_step
            )

        if batch_idx % 50 == 0:
            print(
                f"Epoch [{epoch + 1}/{n_epochs}] "
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.6f} "
                f"Full: {full_loss.item():.6f} "
                f"Embed: {embedding_loss.item():.6f} "
                f"SVDD: {svdd_loss.item():.6f} "
                f"LambdaSVDD: {(lambda_svdd * svdd_loss.item()):.6f} "
                f"Perplexity: {perplexity.item():.4f}"
            )

        global_step += 1

    avg_train_loss = np.mean(train_losses)
    avg_train_full_loss = np.mean(train_full_losses)
    avg_train_embedding_loss = np.mean(train_embedding_losses)
    avg_train_svdd_loss = np.mean(train_svdd_losses)
    avg_train_perplexity = np.mean(train_perplexities)

    # =============================
    # Validation
    # =============================

    model.eval()
    svdd_head.eval()

    val_losses = []
    val_full_losses = []
    val_embedding_losses = []
    val_svdd_losses = []
    val_perplexities = []

    with torch.no_grad():

        for val_idx, x in enumerate(val_loader):

            x = x.to(device)

            (
                loss,
                full_loss,
                embedding_loss,
                svdd_loss,
                perplexity,
                x_hat,
                q_distance_map,
                z_q,
                svdd_map
            ) = compute_losses(
                model,
                svdd_head,
                x,
                center_map
            )

            val_losses.append(loss.item())
            val_full_losses.append(full_loss.item())
            val_embedding_losses.append(embedding_loss.item())
            val_svdd_losses.append(svdd_loss.item())
            val_perplexities.append(perplexity.item())

            if val_idx == 0:

                n_show = min(8, x.size(0))

                full_diff = torch.abs(x - x_hat)

                svdd_map_up = upsample_map(
                    svdd_map,
                    x.shape[-2:]
                )

                svdd_map_vis = normalize_map(
                    svdd_map_up
                )

                full_diff_vis = normalize_map(
                    full_diff
                )

                grid = vutils.make_grid(
                    torch.cat([
                        x[:n_show],
                        x_hat[:n_show],
                        full_diff_vis[:n_show],
                        svdd_map_vis[:n_show]
                    ], dim=0),
                    nrow=n_show,
                    normalize=True
                )

                writer.add_image(
                    "Val/Input_Recon_FullDiff_SpatialSVDD",
                    grid,
                    epoch
                )

    avg_val_loss = np.mean(val_losses)
    avg_val_full_loss = np.mean(val_full_losses)
    avg_val_embedding_loss = np.mean(val_embedding_losses)
    avg_val_svdd_loss = np.mean(val_svdd_losses)
    avg_val_perplexity = np.mean(val_perplexities)

    writer.add_scalar(
        "Epoch/Train_Loss",
        avg_train_loss,
        epoch
    )

    writer.add_scalar(
        "Epoch/Train_Full_Loss",
        avg_train_full_loss,
        epoch
    )

    writer.add_scalar(
        "Epoch/Train_Embedding_Loss",
        avg_train_embedding_loss,
        epoch
    )

    writer.add_scalar(
        "Epoch/Train_SVDD_Loss",
        avg_train_svdd_loss,
        epoch
    )

    writer.add_scalar(
        "Epoch/Train_Perplexity",
        avg_train_perplexity,
        epoch
    )

    writer.add_scalar(
        "Epoch/Val_Loss",
        avg_val_loss,
        epoch
    )

    writer.add_scalar(
        "Epoch/Val_Full_Loss",
        avg_val_full_loss,
        epoch
    )

    writer.add_scalar(
        "Epoch/Val_Embedding_Loss",
        avg_val_embedding_loss,
        epoch
    )

    writer.add_scalar(
        "Epoch/Val_SVDD_Loss",
        avg_val_svdd_loss,
        epoch
    )

    writer.add_scalar(
        "Epoch/Val_Perplexity",
        avg_val_perplexity,
        epoch
    )

    print(
        f"\nEpoch [{epoch + 1}/{n_epochs}] Finished\n"
        f"Train Loss: {avg_train_loss:.6f} | "
        f"Train Full: {avg_train_full_loss:.6f} | "
        f"Train Embed: {avg_train_embedding_loss:.6f} | "
        f"Train SVDD: {avg_train_svdd_loss:.6f} | "
        f"Train Perplexity: {avg_train_perplexity:.4f}\n"
        f"Val Loss: {avg_val_loss:.6f} | "
        f"Val Full: {avg_val_full_loss:.6f} | "
        f"Val Embed: {avg_val_embedding_loss:.6f} | "
        f"Val SVDD: {avg_val_svdd_loss:.6f} | "
        f"Val Perplexity: {avg_val_perplexity:.4f}\n"
    )

    # Save best model
    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        save_checkpoint(
            model,
            svdd_head,
            optimizer,
            epoch + 1,
            avg_val_loss,
            f"{filename}_best"
        )

    # Save every 5 epochs
    if (epoch + 1) % 5 == 0:

        save_checkpoint(
            model,
            svdd_head,
            optimizer,
            epoch + 1,
            avg_val_loss,
            f"{filename}_epoch{epoch + 1}"
        )


# =============================
# Save final model
# =============================

save_checkpoint(
    model,
    svdd_head,
    optimizer,
    n_epochs,
    avg_val_loss,
    f"{filename}_final"
)

writer.close()

print("Stage3 Spatial8 SVDD Training Finished.")
print("Best Val Loss:", best_val_loss)
print("Results saved to:", result_dir)
print("TensorBoard logdir:", f"./runs/{filename}_{run_name}")