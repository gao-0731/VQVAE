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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =============================
# Training Settings
# =============================

batch_size = 16
n_epochs = 100
learning_rate = 5e-5
resize = 256

filename = "vqvae_ct_lung_region_masked_stage4"


# =============================
# VQ-VAE Parameters
# 必须和 Stage2_v2 一致
# =============================

n_hiddens = 128
n_residual_hiddens = 64
n_residual_layers = 1

embedding_dim = 64
n_embeddings = 64
beta = 0.25


# =============================
# Mask Settings
# =============================

mask_ratio = 0.05
block_size = 4
alpha_masked = 0.2

# 用于扩张 lung region，包含 pleural boundary / 贴胸壁区域
lung_dilate_kernel = 21


# =============================
# Paths
# =============================

train_data_dir = "/app/data/train"
val_data_dir = "/app/data/validation"

result_dir = "/app/results"
os.makedirs(result_dir, exist_ok=True)

pretrained_model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage2_lightmask_v2_final.pth"

run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

writer = SummaryWriter(
    log_dir=f"./runs/{filename}_{run_name}"
)

print("Run name:", run_name)
print("Filename:", filename)
print("Pretrained:", pretrained_model_path)
print("TensorBoard:", f"./runs/{filename}_{run_name}")


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
                    self.dicom_files.append(os.path.join(root, file))

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

        # Lung window: [-1000, 400] -> [0, 1]
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


checkpoint = torch.load(
    pretrained_model_path,
    map_location=device
)

if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

print("Loaded pretrained Stage2_v2:", pretrained_model_path)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# =============================
# Lung-region mask helpers
# =============================

def make_lung_region_mask(x):
    """
    x: [B,1,256,256], normalized lung window [0,1]

    return:
        lung_region: [B,1,256,256], float 0/1

    简单规则：
    - 排除纯黑背景
    - 保留肺野和部分软组织边界
    - 再 dilate，包含胸膜/贴胸壁区域
    """

    # lung window 下：
    # background air 接近 0
    # lung parenchyma 大约 0.05~0.45
    # soft tissue/bone 更高
    #
    # 这里阈值故意放宽到 0.65，
    # 是为了包含 pleural boundary / 贴胸壁区域。
    lung = ((x > 0.02) & (x < 0.65)).float()

    pad = lung_dilate_kernel // 2

    lung_dilated = F.max_pool2d(
        lung,
        kernel_size=lung_dilate_kernel,
        stride=1,
        padding=pad
    )

    return lung_dilated


def make_lung_latent_block_mask(
    x,
    latent_h,
    latent_w,
    mask_ratio=0.05,
    block_size=4
):
    """
    x: [B,1,256,256]

    return:
        latent_mask: [B,1,latent_h,latent_w]
        image_mask:  [B,1,256,256]
        lung_region: [B,1,256,256]

    mask 只在 lung_region + boundary 内采样 block
    """

    b = x.size(0)

    lung_region = make_lung_region_mask(x)

    lung_latent = F.interpolate(
        lung_region,
        size=(latent_h, latent_w),
        mode="nearest"
    )

    latent_mask = torch.zeros(
        b,
        1,
        latent_h,
        latent_w,
        device=x.device
    )

    for i in range(b):

        valid = lung_latent[i, 0] > 0.5
        valid_indices = torch.nonzero(valid, as_tuple=False)

        num_valid = valid_indices.size(0)

        if num_valid == 0:
            continue

        target_pixels = int(num_valid * mask_ratio)
        target_pixels = max(target_pixels, block_size * block_size)

        num_blocks = max(
            1,
            target_pixels // (block_size * block_size)
        )

        for _ in range(num_blocks):

            rand_idx = torch.randint(
                0,
                num_valid,
                (1,),
                device=x.device
            ).item()

            cy, cx = valid_indices[rand_idx]

            y0 = int(cy.item()) - block_size // 2
            x0 = int(cx.item()) - block_size // 2

            y0 = max(0, min(y0, latent_h - block_size))
            x0 = max(0, min(x0, latent_w - block_size))

            latent_mask[
                i,
                0,
                y0:y0 + block_size,
                x0:x0 + block_size
            ] = 1.0

    # 保险：只保留 lung region 内的 mask
    latent_mask = latent_mask * lung_latent

    image_mask = F.interpolate(
        latent_mask,
        size=x.shape[-2:],
        mode="nearest"
    )

    return latent_mask, image_mask, lung_region


# =============================
# Forward with lung latent mask
# =============================

def forward_lung_masked(model, x):
    """
    x: [B,1,256,256]

    return:
        embedding_loss
        x_hat
        perplexity
        q_distance_map
        image_mask
        lung_region
    """

    z_e = model.encoder(x)
    z_e = model.pre_quantization_conv(z_e)

    # position bias
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

    latent_h, latent_w = z_e.shape[-2:]

    latent_mask, image_mask, lung_region = make_lung_latent_block_mask(
        x,
        latent_h=latent_h,
        latent_w=latent_w,
        mask_ratio=mask_ratio,
        block_size=block_size
    )

    # mask latent z_e
    z_e_masked = z_e * (1.0 - latent_mask)

    (
        embedding_loss,
        z_q,
        perplexity,
        _,
        _,
        q_distance_map
    ) = model.vector_quantization(z_e_masked)

    x_hat = model.decoder(z_q)

    return (
        embedding_loss,
        x_hat,
        perplexity,
        q_distance_map,
        image_mask,
        lung_region
    )


# =============================
# Loss
# =============================

def compute_losses(model, x):

    (
        embedding_loss,
        x_hat,
        perplexity,
        q_distance_map,
        image_mask,
        lung_region
    ) = forward_lung_masked(model, x)

    full_loss = torch.mean(
        torch.abs(x_hat - x)
    )

    masked_abs = torch.abs(x_hat - x) * image_mask

    masked_loss = masked_abs.sum() / (
        image_mask.sum() + 1e-8
    )

    recon_loss = full_loss + alpha_masked * masked_loss

    loss = recon_loss + embedding_loss

    return (
        loss,
        recon_loss,
        full_loss,
        masked_loss,
        embedding_loss,
        perplexity,
        x_hat,
        q_distance_map,
        image_mask,
        lung_region
    )


# =============================
# Save checkpoint
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
            "alpha_masked": alpha_masked,
            "lung_dilate_kernel": lung_dilate_kernel,

            "pretrained_model_path": pretrained_model_path,
            "filename": filename,
        },
        save_path
    )

    print("Saved:", save_path)


# =============================
# Visualization helpers
# =============================

def normalize_map(x):
    x = x.detach()
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + 1e-8)


def make_vis_grid(x, x_hat, image_mask, lung_region):

    n_show = min(8, x.size(0))

    full_diff = torch.abs(x - x_hat)

    full_diff_vis = normalize_map(full_diff)
    mask_vis = image_mask
    lung_vis = lung_region

    grid = vutils.make_grid(
        torch.cat([
            x[:n_show],
            x_hat[:n_show],
            full_diff_vis[:n_show],
            mask_vis[:n_show],
            lung_vis[:n_show],
        ], dim=0),
        nrow=n_show,
        normalize=True
    )

    return grid


# =============================
# Training
# =============================

best_val_loss = float("inf")
global_step = 0

print("Start Stage4 Lung-region Masked VQ-VAE training...")

for epoch in range(n_epochs):

    # =============================
    # Train
    # =============================

    model.train()

    train_losses = []
    train_recon_losses = []
    train_full_losses = []
    train_masked_losses = []
    train_embedding_losses = []
    train_perplexities = []

    for batch_idx, x in enumerate(train_loader):

        x = x.to(device)

        optimizer.zero_grad()

        (
            loss,
            recon_loss,
            full_loss,
            masked_loss,
            embedding_loss,
            perplexity,
            x_hat,
            q_distance_map,
            image_mask,
            lung_region
        ) = compute_losses(model, x)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        train_recon_losses.append(recon_loss.item())
        train_full_losses.append(full_loss.item())
        train_masked_losses.append(masked_loss.item())
        train_embedding_losses.append(embedding_loss.item())
        train_perplexities.append(perplexity.item())

        writer.add_scalar("Train/Loss", loss.item(), global_step)
        writer.add_scalar("Train/Reconstruction_Loss", recon_loss.item(), global_step)
        writer.add_scalar("Train/Full_Loss", full_loss.item(), global_step)
        writer.add_scalar("Train/Masked_Loss", masked_loss.item(), global_step)
        writer.add_scalar("Train/Embedding_Loss", embedding_loss.item(), global_step)
        writer.add_scalar("Train/Perplexity", perplexity.item(), global_step)

        if global_step % 200 == 0:
            grid = make_vis_grid(
                x,
                x_hat,
                image_mask,
                lung_region
            )

            writer.add_image(
                "Train/Input_Recon_Diff_Mask_Lung",
                grid,
                global_step
            )

        if batch_idx % 50 == 0:
            print(
                f"Epoch [{epoch + 1}/{n_epochs}] "
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.6f} "
                f"Full: {full_loss.item():.6f} "
                f"Masked: {masked_loss.item():.6f} "
                f"Embed: {embedding_loss.item():.6f} "
                f"Perp: {perplexity.item():.4f}"
            )

        global_step += 1

    avg_train_loss = np.mean(train_losses)
    avg_train_recon_loss = np.mean(train_recon_losses)
    avg_train_full_loss = np.mean(train_full_losses)
    avg_train_masked_loss = np.mean(train_masked_losses)
    avg_train_embedding_loss = np.mean(train_embedding_losses)
    avg_train_perplexity = np.mean(train_perplexities)

    # =============================
    # Validation
    # =============================

    model.eval()

    val_losses = []
    val_recon_losses = []
    val_full_losses = []
    val_masked_losses = []
    val_embedding_losses = []
    val_perplexities = []

    with torch.no_grad():

        for val_idx, x in enumerate(val_loader):

            x = x.to(device)

            (
                loss,
                recon_loss,
                full_loss,
                masked_loss,
                embedding_loss,
                perplexity,
                x_hat,
                q_distance_map,
                image_mask,
                lung_region
            ) = compute_losses(model, x)

            val_losses.append(loss.item())
            val_recon_losses.append(recon_loss.item())
            val_full_losses.append(full_loss.item())
            val_masked_losses.append(masked_loss.item())
            val_embedding_losses.append(embedding_loss.item())
            val_perplexities.append(perplexity.item())

            if val_idx == 0:
                grid = make_vis_grid(
                    x,
                    x_hat,
                    image_mask,
                    lung_region
                )

                writer.add_image(
                    "Val/Input_Recon_Diff_Mask_Lung",
                    grid,
                    epoch
                )

    avg_val_loss = np.mean(val_losses)
    avg_val_recon_loss = np.mean(val_recon_losses)
    avg_val_full_loss = np.mean(val_full_losses)
    avg_val_masked_loss = np.mean(val_masked_losses)
    avg_val_embedding_loss = np.mean(val_embedding_losses)
    avg_val_perplexity = np.mean(val_perplexities)

    writer.add_scalar("Epoch/Train_Loss", avg_train_loss, epoch)
    writer.add_scalar("Epoch/Train_Reconstruction_Loss", avg_train_recon_loss, epoch)
    writer.add_scalar("Epoch/Train_Full_Loss", avg_train_full_loss, epoch)
    writer.add_scalar("Epoch/Train_Masked_Loss", avg_train_masked_loss, epoch)
    writer.add_scalar("Epoch/Train_Embedding_Loss", avg_train_embedding_loss, epoch)
    writer.add_scalar("Epoch/Train_Perplexity", avg_train_perplexity, epoch)

    writer.add_scalar("Epoch/Val_Loss", avg_val_loss, epoch)
    writer.add_scalar("Epoch/Val_Reconstruction_Loss", avg_val_recon_loss, epoch)
    writer.add_scalar("Epoch/Val_Full_Loss", avg_val_full_loss, epoch)
    writer.add_scalar("Epoch/Val_Masked_Loss", avg_val_masked_loss, epoch)
    writer.add_scalar("Epoch/Val_Embedding_Loss", avg_val_embedding_loss, epoch)
    writer.add_scalar("Epoch/Val_Perplexity", avg_val_perplexity, epoch)

    print(
        f"\nEpoch [{epoch + 1}/{n_epochs}] Finished\n"
        f"Train Loss: {avg_train_loss:.6f} | "
        f"Train Full: {avg_train_full_loss:.6f} | "
        f"Train Masked: {avg_train_masked_loss:.6f} | "
        f"Train Embed: {avg_train_embedding_loss:.6f} | "
        f"Train Perp: {avg_train_perplexity:.4f}\n"
        f"Val Loss: {avg_val_loss:.6f} | "
        f"Val Full: {avg_val_full_loss:.6f} | "
        f"Val Masked: {avg_val_masked_loss:.6f} | "
        f"Val Embed: {avg_val_embedding_loss:.6f} | "
        f"Val Perp: {avg_val_perplexity:.4f}\n"
    )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss

        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            avg_val_loss,
            f"{filename}_best"
        )

    if (epoch + 1) % 5 == 0:
        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            avg_val_loss,
            f"{filename}_epoch{epoch + 1}"
        )


# =============================
# Save final
# =============================

save_checkpoint(
    model,
    optimizer,
    n_epochs,
    avg_val_loss,
    f"{filename}_final"
)

writer.close()

print("Stage4 Lung-region Masked VQ-VAE training finished.")
print("Best val loss:", best_val_loss)
print("Results saved to:", result_dir)
print("TensorBoard logdir:", f"./runs/{filename}_{run_name}")