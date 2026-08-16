import os
from datetime import datetime

import numpy as np
import torch

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

# 先跑 50 epoch，看 reconstruction 是否稳定
n_epochs = 50

learning_rate = 1e-4

resize = 256
save = True

filename = "vqvae_ct_masked_block_stable"


# =============================
# Model Parameters
# 必须和 VQVAE 一致
# =============================

n_hiddens = 128
n_residual_hiddens = 64
n_residual_layers = 1

embedding_dim = 64
n_embeddings = 64
beta = 0.25


# =============================
# Mask Parameters
# =============================

# 原来 0.25 太大，容易导致 reconstruction 块状崩坏
mask_ratio = 0.10
block_size = 8

# latent 是 64x64
# block_size=8 对应原图大约 32x32 pixel


# =============================
# Paths
# =============================

train_data_dir = "/app/data/train"
val_data_dir = "/app/data/validation"

result_dir = "/app/results"
os.makedirs(result_dir, exist_ok=True)

run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

writer = SummaryWriter(
    log_dir=f"./runs/{filename}_{run_name}"
)

print("Run name:", run_name)
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
        # 当前保持和旧训练一致
        # 如果之后改 HU + lung window，
        # train/eval 必须一起改
        # =============================
        image = np.clip(image, 0, 3072)
        image = image / 3072.0

        image = torch.tensor(image).unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return image


# =============================
# Transform
# 必须和 eval 一致
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
            "mask_ratio": mask_ratio,
            "block_size": block_size,
            "filename": filename,
        },
        save_path
    )

    print("Saved model:", save_path)


# =============================
# Helper: masked L1 loss
# =============================

def masked_l1_loss(x_hat, x, image_mask):
    """
    x_hat:      [B, 1, H, W]
    x:          [B, 1, H, W]
    image_mask: [B, 1, H, W]

    只在被 mask 的区域计算 reconstruction loss
    """

    abs_error = torch.abs(x_hat - x)
    masked_error = abs_error * image_mask

    loss = masked_error.sum() / (
        image_mask.sum() + 1e-8
    )

    return loss


# =============================
# Training
# =============================

best_val_loss = float("inf")
global_step = 0

print("Start training...")

for epoch in range(n_epochs):

    # =============================
    # Train
    # =============================

    model.train()

    train_losses = []
    train_recon_losses = []
    train_masked_losses = []
    train_full_losses = []
    train_embedding_losses = []
    train_perplexities = []

    for batch_idx, x in enumerate(train_loader):

        x = x.to(device)

        optimizer.zero_grad()

        # =============================
        # Forward with block latent mask
        # =============================

        (
            embedding_loss,
            x_hat,
            perplexity,
            q_distance_map,
            image_mask
        ) = model(
            x,
            mask=True,
            mask_ratio=mask_ratio,
            block_size=block_size,
            return_mask=True
        )

        # =============================
        # Reconstruction Loss
        # masked loss 为主
        # full loss 用来稳定整体结构
        # =============================

        masked_loss = masked_l1_loss(
            x_hat,
            x,
            image_mask
        )

        full_loss = torch.mean(
            torch.abs(x_hat - x)
        )

        recon_loss = masked_loss + 0.5 * full_loss

        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        train_recon_losses.append(recon_loss.item())
        train_masked_losses.append(masked_loss.item())
        train_full_losses.append(full_loss.item())
        train_embedding_losses.append(embedding_loss.item())
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
            "Train/Reconstruction_Loss",
            recon_loss.item(),
            global_step
        )

        writer.add_scalar(
            "Train/Masked_Loss",
            masked_loss.item(),
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
            "Train/Perplexity",
            perplexity.item(),
            global_step
        )

        # =============================
        # TensorBoard Images
        # 行顺序：
        # input / reconstruction / mask / masked_diff / full_diff
        # =============================

        if global_step % 200 == 0:

            n_show = min(8, x.size(0))

            masked_diff = torch.abs(x - x_hat) * image_mask
            full_diff = torch.abs(x - x_hat)

            grid = vutils.make_grid(
                torch.cat([
                    x[:n_show],
                    x_hat[:n_show],
                    image_mask[:n_show],
                    masked_diff[:n_show],
                    full_diff[:n_show]
                ], dim=0),
                nrow=n_show,
                normalize=True
            )

            writer.add_image(
                "Train/Input_Recon_Mask_MaskedDiff_FullDiff",
                grid,
                global_step
            )

        if batch_idx % 50 == 0:
            print(
                f"Epoch [{epoch + 1}/{n_epochs}] "
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.6f} "
                f"Recon: {recon_loss.item():.6f} "
                f"Masked: {masked_loss.item():.6f} "
                f"Full: {full_loss.item():.6f} "
                f"Embed: {embedding_loss.item():.6f} "
                f"Perplexity: {perplexity.item():.4f}"
            )

        global_step += 1

    avg_train_loss = np.mean(train_losses)
    avg_train_recon_loss = np.mean(train_recon_losses)
    avg_train_masked_loss = np.mean(train_masked_losses)
    avg_train_full_loss = np.mean(train_full_losses)
    avg_train_embedding_loss = np.mean(train_embedding_losses)
    avg_train_perplexity = np.mean(train_perplexities)

    # =============================
    # Validation
    # =============================

    model.eval()

    val_losses = []
    val_recon_losses = []
    val_masked_losses = []
    val_full_losses = []
    val_embedding_losses = []
    val_perplexities = []

    with torch.no_grad():

        for val_idx, x in enumerate(val_loader):

            x = x.to(device)

            (
                embedding_loss,
                x_hat,
                perplexity,
                q_distance_map,
                image_mask
            ) = model(
                x,
                mask=True,
                mask_ratio=mask_ratio,
                block_size=block_size,
                return_mask=True
            )

            masked_loss = masked_l1_loss(
                x_hat,
                x,
                image_mask
            )

            full_loss = torch.mean(
                torch.abs(x_hat - x)
            )

            recon_loss = masked_loss + 0.5 * full_loss

            loss = recon_loss + embedding_loss

            val_losses.append(loss.item())
            val_recon_losses.append(recon_loss.item())
            val_masked_losses.append(masked_loss.item())
            val_full_losses.append(full_loss.item())
            val_embedding_losses.append(embedding_loss.item())
            val_perplexities.append(perplexity.item())

            if val_idx == 0:

                n_show = min(8, x.size(0))

                masked_diff = torch.abs(x - x_hat) * image_mask
                full_diff = torch.abs(x - x_hat)

                grid = vutils.make_grid(
                    torch.cat([
                        x[:n_show],
                        x_hat[:n_show],
                        image_mask[:n_show],
                        masked_diff[:n_show],
                        full_diff[:n_show]
                    ], dim=0),
                    nrow=n_show,
                    normalize=True
                )

                writer.add_image(
                    "Val/Input_Recon_Mask_MaskedDiff_FullDiff",
                    grid,
                    epoch
                )

    avg_val_loss = np.mean(val_losses)
    avg_val_recon_loss = np.mean(val_recon_losses)
    avg_val_masked_loss = np.mean(val_masked_losses)
    avg_val_full_loss = np.mean(val_full_losses)
    avg_val_embedding_loss = np.mean(val_embedding_losses)
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
        "Epoch/Train_Reconstruction_Loss",
        avg_train_recon_loss,
        epoch
    )

    writer.add_scalar(
        "Epoch/Train_Masked_Loss",
        avg_train_masked_loss,
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
        "Epoch/Val_Reconstruction_Loss",
        avg_val_recon_loss,
        epoch
    )

    writer.add_scalar(
        "Epoch/Val_Masked_Loss",
        avg_val_masked_loss,
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
        "Epoch/Val_Perplexity",
        avg_val_perplexity,
        epoch
    )

    print(
        f"\nEpoch [{epoch + 1}/{n_epochs}] Finished\n"
        f"Train Loss: {avg_train_loss:.6f} | "
        f"Train Recon: {avg_train_recon_loss:.6f} | "
        f"Train Masked: {avg_train_masked_loss:.6f} | "
        f"Train Full: {avg_train_full_loss:.6f} | "
        f"Train Embed: {avg_train_embedding_loss:.6f} | "
        f"Train Perplexity: {avg_train_perplexity:.4f}\n"
        f"Val Loss: {avg_val_loss:.6f} | "
        f"Val Recon: {avg_val_recon_loss:.6f} | "
        f"Val Masked: {avg_val_masked_loss:.6f} | "
        f"Val Full: {avg_val_full_loss:.6f} | "
        f"Val Embed: {avg_val_embedding_loss:.6f} | "
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
    # Save every 10 epochs
    # =============================

    if (epoch + 1) % 10 == 0:
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

print("Training Finished.")
print("Best Val Loss:", best_val_loss)
print("Results saved to:", result_dir)
print("TensorBoard logdir:", f"./runs/{filename}_{run_name}")