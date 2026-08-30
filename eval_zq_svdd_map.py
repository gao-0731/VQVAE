import os
import csv
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.utils as vutils

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
batch_size = 1


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

normal_data_dir = "/app/data/validation"
abnormal_data_dir = "/app/data/temp/abnormal"

# 推荐用 Stage2_v2，不用刚刚 global SVDD 训练后的 Stage3
model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage2_lightmask_v2_final.pth"

# 上一步算出来的 z_q stats
stats_path = "/app/results/zq_svdd_stats_stage2_v2.pth"

save_root = "/app/results/eval_zq_svdd_map_stage2_v2"
os.makedirs(save_root, exist_ok=True)

normal_save_dir = os.path.join(save_root, "normal")
abnormal_save_dir = os.path.join(save_root, "abnormal")

os.makedirs(normal_save_dir, exist_ok=True)
os.makedirs(abnormal_save_dir, exist_ok=True)

csv_path = os.path.join(save_root, "zq_svdd_scores.csv")


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

normal_dataset = DICOMDataset(
    normal_data_dir,
    transform=transform
)

abnormal_dataset = DICOMDataset(
    abnormal_data_dir,
    transform=transform
)

normal_loader = DataLoader(
    normal_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

abnormal_loader = DataLoader(
    abnormal_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print("Normal batches:", len(normal_loader))
print("Abnormal batches:", len(abnormal_loader))


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
# Load z_q SVDD stats
# =============================

stats_ckpt = torch.load(
    stats_path,
    map_location=device
)

center_map = stats_ckpt["center_map"].to(device)
std_map = stats_ckpt["std_map"].to(device)

print("Loaded stats:", stats_path)
print("center_map shape:", center_map.shape)
print("std_map shape:", std_map.shape)
print("center_map mean:", center_map.mean().item())
print("center_map std:", center_map.std().item())
print("std_map mean:", std_map.mean().item())
print("std_map std:", std_map.std().item())


# =============================
# Forward with z_q
# =============================

def forward_with_zq(model, x):
    """
    x: [B,1,256,256]

    returns:
        x_hat: [B,1,256,256]
        q_distance_map: [B,64,64]
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

    x_hat = model.decoder(z_q)

    return x_hat, q_distance_map, z_q, perplexity


# =============================
# Visualization Helpers
# =============================

def normalize_map(x):
    x = x.detach()
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + 1e-8)


def percentile_clip_normalize(x, lower=1, upper=99):
    """
    x: tensor
    用 percentile clip 防止一个极端点把整张图压暗
    """

    x_np = x.detach().cpu().numpy()

    lo = np.percentile(x_np, lower)
    hi = np.percentile(x_np, upper)

    x = torch.clamp(x, min=lo, max=hi)
    x = (x - lo) / (hi - lo + 1e-8)

    return x


def upsample_map(score_map, target_size=(256, 256)):
    """
    score_map: [B,1,64,64] or [B,64,64]
    """

    if score_map.dim() == 3:
        score_map = score_map.unsqueeze(1)

    score_map = F.interpolate(
        score_map,
        size=target_size,
        mode="bilinear",
        align_corners=False
    )

    return score_map


def save_eval_image(
    x,
    x_hat,
    full_diff,
    q_distance_map,
    raw_svdd_map,
    zscore_svdd_map,
    save_path
):
    """
    保存：
    input | recon | full_diff | qdistance | raw_zq_svdd | zscore_zq_svdd
    """

    q_map = q_distance_map.unsqueeze(1)
    q_map = upsample_map(q_map, x.shape[-2:])

    raw_map = upsample_map(raw_svdd_map, x.shape[-2:])
    zscore_map = upsample_map(zscore_svdd_map, x.shape[-2:])

    diff_vis = percentile_clip_normalize(full_diff)
    q_vis = percentile_clip_normalize(q_map)
    raw_vis = percentile_clip_normalize(raw_map)
    zscore_vis = percentile_clip_normalize(zscore_map)

    grid = vutils.make_grid(
        torch.cat([
            x.cpu(),
            x_hat.cpu(),
            diff_vis.cpu(),
            q_vis.cpu(),
            raw_vis.cpu(),
            zscore_vis.cpu()
        ], dim=0),
        nrow=6,
        normalize=True
    )

    vutils.save_image(
        grid,
        save_path
    )


# =============================
# Eval Function
# =============================

def evaluate_loader(
    loader,
    label_name,
    save_dir,
    csv_writer
):

    raw_scores = []
    zscore_scores = []
    full_losses = []
    q_scores = []
    perplexities = []

    with torch.no_grad():

        for idx, batch in enumerate(loader):

            x, paths = batch
            x = x.to(device)

            x_hat, q_distance_map, z_q, perplexity = forward_with_zq(
                model,
                x
            )

            full_diff = torch.abs(x - x_hat)
            full_loss = torch.mean(full_diff).item()

            # =============================
            # Raw position-aware z_q SVDD
            # z_q:       [B,64,64,64]
            # center_map:[1,64,64,64]
            # raw_map:   [B,1,64,64]
            # =============================

            raw_svdd_map = torch.mean(
                (z_q - center_map) ** 2,
                dim=1,
                keepdim=True
            )

            # =============================
            # Z-score normalized SVDD
            # =============================

            zscore_svdd_map = torch.mean(
                ((z_q - center_map) / (std_map + 1e-6)) ** 2,
                dim=1,
                keepdim=True
            )

            # slice-level summary score
            raw_score = raw_svdd_map.mean().item()
            zscore_score = zscore_svdd_map.mean().item()
            q_score = q_distance_map.mean().item()

            raw_scores.append(raw_score)
            zscore_scores.append(zscore_score)
            q_scores.append(q_score)
            full_losses.append(full_loss)
            perplexities.append(perplexity.item())

            dicom_path = paths[0]
            case_name = os.path.basename(dicom_path)

            csv_writer.writerow([
                label_name,
                idx,
                case_name,
                dicom_path,
                raw_score,
                zscore_score,
                q_score,
                full_loss,
                perplexity.item()
            ])

            if idx < 300:
                image_save_path = os.path.join(
                    save_dir,
                    f"{idx:04d}_{case_name}.png"
                )

                save_eval_image(
                    x,
                    x_hat,
                    full_diff,
                    q_distance_map,
                    raw_svdd_map,
                    zscore_svdd_map,
                    image_save_path
                )

            if idx % 50 == 0:
                print(
                    f"[{label_name}] "
                    f"{idx}/{len(loader)} | "
                    f"RawZQ: {raw_score:.6f} | "
                    f"ZScoreZQ: {zscore_score:.6f} | "
                    f"QDist: {q_score:.6f} | "
                    f"FullDiff: {full_loss:.6f} | "
                    f"Perplexity: {perplexity.item():.4f}"
                )

    raw_scores = np.array(raw_scores)
    zscore_scores = np.array(zscore_scores)
    q_scores = np.array(q_scores)
    full_losses = np.array(full_losses)
    perplexities = np.array(perplexities)

    print(f"\n==== {label_name} Summary ====")
    print("Num samples:", len(raw_scores))

    print("Raw z_q SVDD mean:", raw_scores.mean())
    print("Raw z_q SVDD std:", raw_scores.std())
    print("Raw z_q SVDD min:", raw_scores.min())
    print("Raw z_q SVDD max:", raw_scores.max())

    print("ZScore z_q SVDD mean:", zscore_scores.mean())
    print("ZScore z_q SVDD std:", zscore_scores.std())
    print("ZScore z_q SVDD min:", zscore_scores.min())
    print("ZScore z_q SVDD max:", zscore_scores.max())

    print("QDist mean:", q_scores.mean())
    print("QDist std:", q_scores.std())

    print("Full diff mean:", full_losses.mean())
    print("Full diff std:", full_losses.std())

    print("Perplexity mean:", perplexities.mean())
    print("Perplexity std:", perplexities.std())

    return raw_scores, zscore_scores, q_scores, full_losses, perplexities


# =============================
# Run Eval
# =============================

with open(csv_path, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "label",
        "index",
        "filename",
        "path",
        "raw_zq_svdd_score",
        "zscore_zq_svdd_score",
        "qdistance_score",
        "full_diff",
        "perplexity"
    ])

    normal_raw, normal_zscore, normal_q, normal_diff, normal_perp = evaluate_loader(
        normal_loader,
        "normal",
        normal_save_dir,
        writer
    )

    abnormal_raw, abnormal_zscore, abnormal_q, abnormal_diff, abnormal_perp = evaluate_loader(
        abnormal_loader,
        "abnormal",
        abnormal_save_dir,
        writer
    )


# =============================
# Final Comparison
# =============================

print("\n==============================")
print("Final Comparison")
print("==============================")

print(
    f"Normal RawZQ:   mean={normal_raw.mean():.6f}, "
    f"std={normal_raw.std():.6f}, "
    f"min={normal_raw.min():.6f}, "
    f"max={normal_raw.max():.6f}"
)

print(
    f"Abnormal RawZQ: mean={abnormal_raw.mean():.6f}, "
    f"std={abnormal_raw.std():.6f}, "
    f"min={abnormal_raw.min():.6f}, "
    f"max={abnormal_raw.max():.6f}"
)

print(
    f"Normal ZScoreZQ:   mean={normal_zscore.mean():.6f}, "
    f"std={normal_zscore.std():.6f}, "
    f"min={normal_zscore.min():.6f}, "
    f"max={normal_zscore.max():.6f}"
)

print(
    f"Abnormal ZScoreZQ: mean={abnormal_zscore.mean():.6f}, "
    f"std={abnormal_zscore.std():.6f}, "
    f"min={abnormal_zscore.min():.6f}, "
    f"max={abnormal_zscore.max():.6f}"
)

print(
    f"Normal QDist:   mean={normal_q.mean():.6f}, "
    f"std={normal_q.std():.6f}"
)

print(
    f"Abnormal QDist: mean={abnormal_q.mean():.6f}, "
    f"std={abnormal_q.std():.6f}"
)

print(
    f"Normal FullDiff:   mean={normal_diff.mean():.6f}, "
    f"std={normal_diff.std():.6f}"
)

print(
    f"Abnormal FullDiff: mean={abnormal_diff.mean():.6f}, "
    f"std={abnormal_diff.std():.6f}"
)

print("CSV saved to:", csv_path)
print("Images saved to:", save_root)


# =============================
# Simple threshold checks
# =============================

raw_threshold = normal_raw.mean() + 3.0 * normal_raw.std()
zscore_threshold = normal_zscore.mean() + 3.0 * normal_zscore.std()

raw_detected = abnormal_raw > raw_threshold
raw_false_positive = normal_raw > raw_threshold

zscore_detected = abnormal_zscore > zscore_threshold
zscore_false_positive = normal_zscore > zscore_threshold

print("\n==== Threshold: normal mean + 3std ====")

print("RawZQ threshold:", raw_threshold)
print(
    "RawZQ normal false positive:",
    raw_false_positive.sum(),
    "/",
    len(raw_false_positive)
)
print(
    "RawZQ abnormal detected:",
    raw_detected.sum(),
    "/",
    len(raw_detected)
)

print("ZScoreZQ threshold:", zscore_threshold)
print(
    "ZScoreZQ normal false positive:",
    zscore_false_positive.sum(),
    "/",
    len(zscore_false_positive)
)
print(
    "ZScoreZQ abnormal detected:",
    zscore_detected.sum(),
    "/",
    len(zscore_detected)
)