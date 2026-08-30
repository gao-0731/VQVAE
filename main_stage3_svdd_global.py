import os
from datetime import datetime

import numpy as np
import torch
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

# SVDD fine-tuning 建议稍微小一点
learning_rate = 5e-5

resize = 256

filename = "vqvae_ct_lungwindow_stage3_global_svdd"


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


# =============================
# Paths
# =============================

train_data_dir = "/app/data/train"
val_data_dir = "/app/data/validation"

result_dir = "/app/results"
os.makedirs(result_dir, exist_ok=True)

# Stage2_v2 模型
pretrained_model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage2_lightmask_v2_final.pth"

# 你刚刚计算出来的 center
# 如果你的文件名不同，改这里
center_path = "/app/results/svdd_center_global_stage2_v2.pth"
# center_path = "/app/results/svdd_center_global_stage2_v2_train.pth"
# center_path = "/app/results/svdd_center_global_stage2_v2_val.pth"

run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

writer = SummaryWriter(
    log_dir=f"./runs/{filename}_{run_name}"
)

print("Run name:", run_name)
print("Filename:", filename)
print("Pretrained model:", pretrained_model_path)
print("Center path:", center_path)
print("TensorBoard logdir:", f"./runs/{filename}_{run_name}")


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


# =============================
# Load Stage2_v2 checkpoint
# =============================

checkpoint = torch.load(
    pretrained_model_path,
    map_location=device
)

if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

print("Loaded pretrained model:", pretrained_model_path)


# =============================
# Load SVDD center
# =============================

center_ckpt = torch.load(
    center_path,
    map_location=device
)

center = center_ckpt["center"].to(device)

# center: [64]
print("Loaded center:", center_path)
print("Center shape:", center.shape)
print("Center mean:", center.mean().item())
print("Center std:", center.std().item())


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# =============================
# Helper: save checkpoint
# =============================

def save_checkpoint(model, optimizer, epoch, loss, name):

    save_path = os.path.join(
        result_dir,
        f"vqvae_data_{name}.pth"
    )

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss,

            "n_hiddens": n_hiddens,
            "n_residual_hiddens": n_residual_hiddens,
            "n_residual_layers": n_residual_layers,

            "embedding_dim": embedding_dim,
            "n_embeddings": n_embeddings,
            "beta": beta,

            "lambda_svdd": lambda_svdd,
            "center_path": center_path,
            "pretrained_model_path": pretrained_model_path,
            "filename": filename,
        },
        save_path
    )

    print("Saved model:", save_path)


# =============================
# Helper: forward with z_q
# =============================

def forward_with_zq(model, x):
    """
    x: [B,1,256,256]

    returns:
        embedding_loss
        x_hat
        perplexity
        q_distance_map
        z_q: [B,C,64,64]
    """

    # Encoder
    z_e = model.encoder(x)

    # 1x1 projection
    z_e = model.pre_quantization_conv(z_e)

    # Position bias
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

    # Vector quantization
    (
        embedding_loss,
        z_q,
        perplexity,
        _,
        _,
        q_distance_map
    ) = model.vector_quantization(z_e)

    # Decoder
    x_hat = model.decoder(z_q)

    return (
        embedding_loss,
        x_hat,
        perplexity,
        q_distance_map,
        z_q
    )


# =============================
# Helper: compute losses
# =============================

def compute_losses(model, x, center):

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

    # =============================
    # Full reconstruction loss
    # =============================

    full_loss = torch.mean(
        torch.abs(x_hat - x)
    )

    # =============================
    # Global SVDD loss
    # z_global: [B, C]
    # center:   [C]
    # =============================

    z_global = z_q.mean(
        dim=[2, 3]
    )

    svdd_loss = torch.mean(
        torch.sum(
            (z_global - center) ** 2,
            dim=1
        )
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
        z_q
    )


# =============================
# Training
# =============================

best_val_loss = float("inf")
global_step = 0

print("Start Stage3 Global SVDD training...")

for epoch in range(n_epochs):

    # =============================
    # Train
    # =============================

    model.train()

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
            z_q
        ) = compute_losses(
            model,
            x,
            center
        )

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        train_full_losses.append(full_loss.item())
        train_embedding_losses.append(embedding_loss.item())
        train_svdd_losses.append(svdd_loss.item())
        train_perplexities.append(perplexity.item())

        # =============================
        # TensorBoard Scalars
        # =============================

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

        # =============================
        # TensorBoard Images
        # =============================

        if global_step % 200 == 0:

            n_show = min(8, x.size(0))

            full_diff = torch.abs(x - x_hat)

            grid = vutils.make_grid(
                torch.cat([
                    x[:n_show],
                    x_hat[:n_show],
                    full_diff[:n_show]
                ], dim=0),
                nrow=n_show,
                normalize=True
            )

            writer.add_image(
                "Train/Input_Recon_FullDiff",
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
                z_q
            ) = compute_losses(
                model,
                x,
                center
            )

            val_losses.append(loss.item())
            val_full_losses.append(full_loss.item())
            val_embedding_losses.append(embedding_loss.item())
            val_svdd_losses.append(svdd_loss.item())
            val_perplexities.append(perplexity.item())

            if val_idx == 0:

                n_show = min(8, x.size(0))

                full_diff = torch.abs(x - x_hat)

                grid = vutils.make_grid(
                    torch.cat([
                        x[:n_show],
                        x_hat[:n_show],
                        full_diff[:n_show]
                    ], dim=0),
                    nrow=n_show,
                    normalize=True
                )

                writer.add_image(
                    "Val/Input_Recon_FullDiff",
                    grid,
                    epoch
                )

    avg_val_loss = np.mean(val_losses)
    avg_val_full_loss = np.mean(val_full_losses)
    avg_val_embedding_loss = np.mean(val_embedding_losses)
    avg_val_svdd_loss = np.mean(val_svdd_losses)
    avg_val_perplexity = np.mean(val_perplexities)

    # =============================
    # TensorBoard Epoch Scalars
    # =============================

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

    # =============================
    # Save best model
    # =============================

    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            avg_val_loss,
            f"{filename}_best"
        )

    # =============================
    # Save every 5 epochs
    # =============================

    if (epoch + 1) % 5 == 0:

        save_checkpoint(
            model,
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
    optimizer,
    n_epochs,
    avg_val_loss,
    f"{filename}_final"
)

writer.close()

print("Stage3 Global SVDD Training Finished.")
print("Best Val Loss:", best_val_loss)
print("Results saved to:", result_dir)
print("TensorBoard logdir:", f"./runs/{filename}_{run_name}")