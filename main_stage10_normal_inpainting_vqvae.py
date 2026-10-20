import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils as vutils

import pydicom
import numpy as np
from PIL import Image

from pytorch_msssim import ssim

from models.vqvae import VQVAE


# ============================================================
# Settings
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_dir = "/app/data/train"
val_dir = "/app/data/validation"

save_root = "/app/results"
os.makedirs(save_root, exist_ok=True)

run_name = "vqvae_ct_lungwindow_stage10_normal_inpainting"

# TensorBoard
log_dir = os.path.join("/app/runs", run_name)
writer = SummaryWriter(log_dir=log_dir)
print("TensorBoard log dir:", log_dir)

# pretrained Stage1 full VQ-VAE
pretrained_model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage1_full_final.pth"

# output checkpoints
best_model_path = os.path.join(save_root, f"{run_name}_best.pth")
final_model_path = os.path.join(save_root, f"{run_name}_final.pth")

# visualization
vis_dir = os.path.join(save_root, run_name + "_vis")
os.makedirs(vis_dir, exist_ok=True)

resize = 256

n_epochs = 50
batch_size = 16
learning_rate = 5e-5

num_workers = 4

# mask setting
block_size_list = [16, 24, 32]
num_masks_per_image = 1
fill_mode = "local_mean"     # "local_mean" or "constant"
fill_value = 0.08
min_lung_ratio = 0.25
max_try = 50

# loss weights
lambda_ssim = 0.1
lambda_masked = 2.0

# validation / save
val_interval = 1
save_vis_interval = 1


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
# Dataset
# ============================================================

def collect_dicom_paths(root_dir):
    paths = []

    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".dcm"):
                paths.append(os.path.join(root, f))

    paths = sorted(paths)
    return paths


def load_dicom_lung_window(dicom_path, resize=256):
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

    image = torch.tensor(image).unsqueeze(0)

    return image


class CTNormalDataset(Dataset):
    def __init__(self, root_dir, resize=256):
        self.paths = collect_dicom_paths(root_dir)
        self.resize = resize

        print(f"Dataset: {root_dir}")
        print("Number of DICOM files:", len(self.paths))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        x = load_dicom_lung_window(path, resize=self.resize)

        return x, path


# ============================================================
# Mask functions
# ============================================================

def make_lung_mask(x):
    """
    x: [B,1,256,256]
    lung mask based on normalized lung-window intensity.
    """

    lung = ((x > 0.02) & (x < 0.45)).float()

    # small smoothing
    lung = F.max_pool2d(
        lung,
        kernel_size=5,
        stride=1,
        padding=2
    )

    lung = -F.max_pool2d(
        -lung,
        kernel_size=5,
        stride=1,
        padding=2
    )

    lung = (lung > 0.5).float()

    return lung


def get_local_mean_fill(x, y1, y2, x1, x2, pad=8):
    """
    x: [1,H,W]
    return scalar local mean around block.
    """

    _, h, w = x.shape

    yy1 = max(0, y1 - pad)
    yy2 = min(h, y2 + pad)
    xx1 = max(0, x1 - pad)
    xx2 = min(w, x2 + pad)

    region = x[:, yy1:yy2, xx1:xx2].clone()

    inner_y1 = y1 - yy1
    inner_y2 = y2 - yy1
    inner_x1 = x1 - xx1
    inner_x2 = x2 - xx1

    valid = torch.ones_like(region)
    valid[:, inner_y1:inner_y2, inner_x1:inner_x2] = 0.0

    if valid.sum() < 1:
        return torch.tensor(fill_value, dtype=x.dtype, device=x.device)

    mean_value = (region * valid).sum() / (valid.sum() + 1e-8)

    return mean_value


def apply_random_lung_block_mask(x):
    """
    x: [B,1,256,256]

    return:
        x_masked: [B,1,256,256]
        mask:     [B,1,256,256]
    """

    b, c, h, w = x.shape

    lung_mask = make_lung_mask(x)

    x_masked = x.clone()
    mask = torch.zeros_like(x)

    for bi in range(b):
        for _ in range(num_masks_per_image):

            found = False

            for _try in range(max_try):
                block_size = random.choice(block_size_list)

                y1 = random.randint(0, h - block_size)
                x1 = random.randint(0, w - block_size)
                y2 = y1 + block_size
                x2 = x1 + block_size

                lung_ratio = lung_mask[bi:bi+1, :, y1:y2, x1:x2].mean().item()

                if lung_ratio >= min_lung_ratio:
                    found = True
                    break

            if not found:
                block_size = random.choice(block_size_list)

                y1 = random.randint(0, h - block_size)
                x1 = random.randint(0, w - block_size)
                y2 = y1 + block_size
                x2 = x1 + block_size

            if fill_mode == "local_mean":
                fill = get_local_mean_fill(
                    x[bi],
                    y1,
                    y2,
                    x1,
                    x2,
                    pad=8
                )
            elif fill_mode == "constant":
                fill = torch.tensor(
                    fill_value,
                    dtype=x.dtype,
                    device=x.device
                )
            else:
                raise ValueError(f"Unknown fill_mode: {fill_mode}")

            x_masked[bi, :, y1:y2, x1:x2] = fill
            mask[bi, :, y1:y2, x1:x2] = 1.0

    return x_masked, mask


# ============================================================
# Loss
# ============================================================

def compute_inpainting_loss(x_clean, recon, mask, embedding_loss):
    """
    x_clean: target clean normal image
    recon:   model output from masked image
    """

    full_l1 = F.l1_loss(recon, x_clean)

    ssim_loss = 1.0 - ssim(
        recon,
        x_clean,
        data_range=1.0,
        size_average=True
    )

    masked_l1 = (
        torch.abs(recon - x_clean) * mask
    ).sum() / (mask.sum() + 1e-8)

    total_loss = (
        full_l1
        + lambda_ssim * ssim_loss
        + lambda_masked * masked_l1
        + embedding_loss
    )

    loss_dict = {
        "total_loss": total_loss,
        "full_l1": full_l1.detach(),
        "ssim_loss": ssim_loss.detach(),
        "masked_l1": masked_l1.detach(),
        "embedding_loss": embedding_loss.detach(),
    }

    return total_loss, loss_dict


# ============================================================
# Model forward
# ============================================================

def forward_vqvae(model, x):
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

    return recon, embedding_loss, perplexity, q_distance_map


# ============================================================
# Visualization
# ============================================================

def make_visualization_grid(x_clean, x_masked, recon, mask, max_images=8):
    """
    columns:
    clean | masked | mask | recon | diff
    """

    x_clean = x_clean[:max_images].detach().cpu()
    x_masked = x_masked[:max_images].detach().cpu()
    recon = recon[:max_images].detach().cpu()
    mask = mask[:max_images].detach().cpu()

    diff = torch.abs(x_clean - recon)
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

    grid_items = []

    for i in range(x_clean.shape[0]):
        grid_items.extend([
            x_clean[i:i+1],
            x_masked[i:i+1],
            mask[i:i+1],
            recon[i:i+1],
            diff[i:i+1],
        ])

    grid = vutils.make_grid(
        torch.cat(grid_items, dim=0),
        nrow=5,
        normalize=True
    )

    return grid


def save_visualization(x_clean, x_masked, recon, mask, save_path):
    grid = make_visualization_grid(
        x_clean,
        x_masked,
        recon,
        mask,
        max_images=8
    )

    vutils.save_image(grid, save_path)

    return grid


# ============================================================
# Load datasets
# ============================================================

train_dataset = CTNormalDataset(train_dir, resize=resize)
val_dataset = CTNormalDataset(val_dir, resize=resize)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True if num_workers > 0 else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True if num_workers > 0 else False
)


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

if pretrained_model_path is not None and os.path.exists(pretrained_model_path):
    checkpoint = torch.load(pretrained_model_path, map_location=device)

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    print("Loaded pretrained model:", pretrained_model_path)
else:
    print("No pretrained model loaded. Training from scratch.")

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)

best_val_loss = float("inf")


# ============================================================
# Training loop
# ============================================================

print()
print("=" * 80)
print("Start Stage10 normal inpainting training")
print("Run name:", run_name)
print("TensorBoard:", log_dir)
print("block_size_list:", block_size_list)
print("fill_mode:", fill_mode)
print("lambda_masked:", lambda_masked)
print("=" * 80)

global_step = 0

try:
    for epoch in range(1, n_epochs + 1):
        model.train()

        train_total = 0.0
        train_full_l1 = 0.0
        train_masked_l1 = 0.0
        train_ssim = 0.0
        train_embed = 0.0
        train_perplexity = 0.0

        start_time = time.time()

        for step, (x_clean, paths) in enumerate(train_loader):
            x_clean = x_clean.to(device, non_blocking=True)

            x_masked, mask = apply_random_lung_block_mask(x_clean)

            recon, embedding_loss, perplexity, q_distance_map = forward_vqvae(
                model,
                x_masked
            )

            loss, loss_dict = compute_inpainting_loss(
                x_clean,
                recon,
                mask,
                embedding_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total += loss_dict["total_loss"].item()
            train_full_l1 += loss_dict["full_l1"].item()
            train_masked_l1 += loss_dict["masked_l1"].item()
            train_ssim += loss_dict["ssim_loss"].item()
            train_embed += loss_dict["embedding_loss"].item()
            train_perplexity += perplexity.item()

            global_step += 1

            # Step-level TensorBoard logging
            if global_step % 50 == 0:
                writer.add_scalar(
                    "Step/Train_Total_Loss",
                    loss_dict["total_loss"].item(),
                    global_step
                )
                writer.add_scalar(
                    "Step/Train_Masked_L1",
                    loss_dict["masked_l1"].item(),
                    global_step
                )
                writer.add_scalar(
                    "Step/Train_Perplexity",
                    perplexity.item(),
                    global_step
                )

            if (step + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch}/{n_epochs}] "
                    f"Step [{step+1}/{len(train_loader)}] "
                    f"Loss={loss_dict['total_loss'].item():.6f} "
                    f"FullL1={loss_dict['full_l1'].item():.6f} "
                    f"MaskedL1={loss_dict['masked_l1'].item():.6f} "
                    f"SSIMLoss={loss_dict['ssim_loss'].item():.6f} "
                    f"Embed={loss_dict['embedding_loss'].item():.6f} "
                    f"Perp={perplexity.item():.4f}"
                )

        n_train = len(train_loader)

        train_total /= n_train
        train_full_l1 /= n_train
        train_masked_l1 /= n_train
        train_ssim /= n_train
        train_embed /= n_train
        train_perplexity /= n_train

        elapsed = time.time() - start_time

        print()
        print(
            f"[Train] Epoch {epoch}/{n_epochs} "
            f"Loss={train_total:.6f} "
            f"FullL1={train_full_l1:.6f} "
            f"MaskedL1={train_masked_l1:.6f} "
            f"SSIMLoss={train_ssim:.6f} "
            f"Embed={train_embed:.6f} "
            f"Perp={train_perplexity:.4f} "
            f"Time={elapsed:.1f}s"
        )

        # Epoch-level TensorBoard logging
        writer.add_scalar("Train/Total_Loss", train_total, epoch)
        writer.add_scalar("Train/Full_L1", train_full_l1, epoch)
        writer.add_scalar("Train/Masked_L1", train_masked_l1, epoch)
        writer.add_scalar("Train/SSIM_Loss", train_ssim, epoch)
        writer.add_scalar("Train/Embedding_Loss", train_embed, epoch)
        writer.add_scalar("Train/Perplexity", train_perplexity, epoch)

        # ========================================================
        # Validation
        # ========================================================

        if epoch % val_interval == 0:
            model.eval()

            val_total = 0.0
            val_full_l1 = 0.0
            val_masked_l1 = 0.0
            val_ssim = 0.0
            val_embed = 0.0
            val_perplexity = 0.0

            last_vis = None

            with torch.no_grad():
                for step, (x_clean, paths) in enumerate(val_loader):
                    x_clean = x_clean.to(device, non_blocking=True)

                    x_masked, mask = apply_random_lung_block_mask(x_clean)

                    recon, embedding_loss, perplexity, q_distance_map = forward_vqvae(
                        model,
                        x_masked
                    )

                    loss, loss_dict = compute_inpainting_loss(
                        x_clean,
                        recon,
                        mask,
                        embedding_loss
                    )

                    val_total += loss_dict["total_loss"].item()
                    val_full_l1 += loss_dict["full_l1"].item()
                    val_masked_l1 += loss_dict["masked_l1"].item()
                    val_ssim += loss_dict["ssim_loss"].item()
                    val_embed += loss_dict["embedding_loss"].item()
                    val_perplexity += perplexity.item()

                    if step == 0:
                        last_vis = (
                            x_clean.detach(),
                            x_masked.detach(),
                            recon.detach(),
                            mask.detach()
                        )

            n_val = len(val_loader)

            val_total /= n_val
            val_full_l1 /= n_val
            val_masked_l1 /= n_val
            val_ssim /= n_val
            val_embed /= n_val
            val_perplexity /= n_val

            print(
                f"[Val]   Epoch {epoch}/{n_epochs} "
                f"Loss={val_total:.6f} "
                f"FullL1={val_full_l1:.6f} "
                f"MaskedL1={val_masked_l1:.6f} "
                f"SSIMLoss={val_ssim:.6f} "
                f"Embed={val_embed:.6f} "
                f"Perp={val_perplexity:.4f}"
            )

            writer.add_scalar("Val/Total_Loss", val_total, epoch)
            writer.add_scalar("Val/Full_L1", val_full_l1, epoch)
            writer.add_scalar("Val/Masked_L1", val_masked_l1, epoch)
            writer.add_scalar("Val/SSIM_Loss", val_ssim, epoch)
            writer.add_scalar("Val/Embedding_Loss", val_embed, epoch)
            writer.add_scalar("Val/Perplexity", val_perplexity, epoch)

            if epoch % save_vis_interval == 0 and last_vis is not None:
                x_clean_vis, x_masked_vis, recon_vis, mask_vis = last_vis

                vis_path = os.path.join(
                    vis_dir,
                    f"epoch_{epoch:03d}_val_inpainting.png"
                )

                tb_grid = save_visualization(
                    x_clean_vis,
                    x_masked_vis,
                    recon_vis,
                    mask_vis,
                    vis_path
                )

                writer.add_image(
                    "Val/clean_mask_recon_diff",
                    tb_grid,
                    epoch
                )

                print("Saved visualization:", vis_path)

            # save best
            if val_total < best_val_loss:
                best_val_loss = val_total

                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_total,
                        "run_name": run_name,
                        "block_size_list": block_size_list,
                        "fill_mode": fill_mode,
                        "lambda_masked": lambda_masked,
                        "pretrained_model_path": pretrained_model_path,
                    },
                    best_model_path
                )

                print("Saved best model:", best_model_path)

        # save periodic checkpoint
        if epoch % 10 == 0:
            ckpt_path = os.path.join(
                save_root,
                f"{run_name}_epoch_{epoch:03d}.pth"
            )

            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "run_name": run_name,
                    "block_size_list": block_size_list,
                    "fill_mode": fill_mode,
                    "lambda_masked": lambda_masked,
                    "pretrained_model_path": pretrained_model_path,
                },
                ckpt_path
            )

            print("Saved checkpoint:", ckpt_path)

        writer.flush()

    # ============================================================
    # Save final model
    # ============================================================

    torch.save(
        {
            "model": model.state_dict(),
            "epoch": n_epochs,
            "run_name": run_name,
            "block_size_list": block_size_list,
            "fill_mode": fill_mode,
            "lambda_masked": lambda_masked,
            "pretrained_model_path": pretrained_model_path,
        },
        final_model_path
    )

    print()
    print("=" * 80)
    print("Training finished.")
    print("Best val loss:", best_val_loss)
    print("Best model:", best_model_path)
    print("Final model:", final_model_path)
    print("Visualization dir:", vis_dir)
    print("TensorBoard log dir:", log_dir)
    print("=" * 80)

finally:
    writer.close()