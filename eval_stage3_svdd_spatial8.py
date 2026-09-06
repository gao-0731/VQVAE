import os
import csv
import numpy as np

import torch
import torch.nn as nn
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
# 必须和 Stage2_v2 / spatial8 training 一致
# =============================

n_hiddens = 128
n_residual_hiddens = 64
n_residual_layers = 1

embedding_dim = 64
n_embeddings = 64
beta = 0.25

svdd_hidden_dim = 128


# =============================
# Paths
# =============================

normal_data_dir = "/app/data/temp/normal"
abnormal_data_dir = "/app/data/temp/abnormal"

# spatial8 训练后的模型
model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage3_spatial8_svdd_final.pth"

# 如果 final 重建变差，可以改成 best
# model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage3_spatial8_svdd_best.pth"

# spatial8 center
center_path = "/app/results/svdd_center_spatial8_stage2_v2.pth"

save_root = "/app/results/eval_stage3_spatial8_svdd"
os.makedirs(save_root, exist_ok=True)

normal_save_dir = os.path.join(save_root, "normal")
abnormal_save_dir = os.path.join(save_root, "abnormal")

os.makedirs(normal_save_dir, exist_ok=True)
os.makedirs(abnormal_save_dir, exist_ok=True)

csv_path = os.path.join(save_root, "spatial8_svdd_scores.csv")


# =============================
# SVDD Head 8x8
# 必须和训练时完全一致
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
            nn.Conv2d(
                in_dim,
                hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(0.1, inplace=True),

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
# Load Model
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
# Load SVDD Head
# =============================

svdd_head = SVDDHead8x8(
    in_dim=embedding_dim,
    hidden_dim=svdd_hidden_dim
).to(device)

if "svdd_head" in checkpoint:
    svdd_head.load_state_dict(checkpoint["svdd_head"])
    print("Loaded trained svdd_head from checkpoint.")
else:
    raise KeyError("checkpoint does not contain 'svdd_head'. Please check model_path.")

svdd_head.eval()


# =============================
# Load Center Map
# =============================

center_ckpt = torch.load(
    center_path,
    map_location=device
)

center_map = center_ckpt["center_map"].to(device)

print("Loaded center:", center_path)
print("center_map shape:", center_map.shape)
print("center_map mean:", center_map.mean().item())
print("center_map std:", center_map.std().item())


# =============================
# Forward with z_q
# =============================

def forward_with_zq(model, x):
    """
    return:
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

def percentile_clip_normalize(x, lower=1, upper=99):
    x_np = x.detach().cpu().numpy()

    lo = np.percentile(x_np, lower)
    hi = np.percentile(x_np, upper)

    x = torch.clamp(x, min=lo, max=hi)
    x = (x - lo) / (hi - lo + 1e-8)

    return x


def upsample_map(score_map, target_size=(256, 256)):
    if score_map.dim() == 3:
        score_map = score_map.unsqueeze(1)

    return F.interpolate(
        score_map,
        size=target_size,
        mode="bilinear",
        align_corners=False
    )


def save_eval_image(
    x,
    x_hat,
    full_diff,
    q_distance_map,
    spatial_svdd_map,
    save_path
):
    """
    保存：
    input | recon | full_diff | qdistance | spatial8_svdd
    """

    q_map = q_distance_map.unsqueeze(1)
    q_map = upsample_map(q_map, x.shape[-2:])

    svdd_up = upsample_map(spatial_svdd_map, x.shape[-2:])

    diff_vis = percentile_clip_normalize(full_diff)
    q_vis = percentile_clip_normalize(q_map)
    svdd_vis = percentile_clip_normalize(svdd_up)

    grid = vutils.make_grid(
        torch.cat([
            x.cpu(),
            x_hat.cpu(),
            diff_vis.cpu(),
            q_vis.cpu(),
            svdd_vis.cpu()
        ], dim=0),
        nrow=5,
        normalize=True
    )

    vutils.save_image(
        grid,
        save_path
    )


# =============================
# Score Helpers
# =============================

def compute_topk_scores(spatial_svdd_map):
    """
    spatial_svdd_map: [B,1,8,8]

    return:
        mean_score
        max_score
        top3_score
        top5_score
        top10_score
    """

    b = spatial_svdd_map.size(0)

    flat = spatial_svdd_map.view(b, -1)

    mean_score = flat.mean(dim=1)
    max_score = flat.max(dim=1).values

    top3_score = torch.topk(flat, k=3, dim=1).values.mean(dim=1)
    top5_score = torch.topk(flat, k=5, dim=1).values.mean(dim=1)
    top10_score = torch.topk(flat, k=10, dim=1).values.mean(dim=1)

    return (
        mean_score,
        max_score,
        top3_score,
        top5_score,
        top10_score
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

    mean_scores = []
    max_scores = []
    top3_scores = []
    top5_scores = []
    top10_scores = []

    q_scores = []
    full_losses = []
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
            # Spatial8 SVDD map
            # =============================

            svdd_feat = svdd_head(z_q)

            spatial_svdd_map = torch.mean(
                (svdd_feat - center_map) ** 2,
                dim=1,
                keepdim=True
            )  # [B,1,8,8]

            (
                mean_score,
                max_score,
                top3_score,
                top5_score,
                top10_score
            ) = compute_topk_scores(spatial_svdd_map)

            mean_value = mean_score.item()
            max_value = max_score.item()
            top3_value = top3_score.item()
            top5_value = top5_score.item()
            top10_value = top10_score.item()

            q_score = q_distance_map.mean().item()

            mean_scores.append(mean_value)
            max_scores.append(max_value)
            top3_scores.append(top3_value)
            top5_scores.append(top5_value)
            top10_scores.append(top10_value)

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
                mean_value,
                max_value,
                top3_value,
                top5_value,
                top10_value,
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
                    spatial_svdd_map,
                    image_save_path
                )

            if idx % 50 == 0:
                print(
                    f"[{label_name}] "
                    f"{idx}/{len(loader)} | "
                    f"Mean: {mean_value:.6f} | "
                    f"Max: {max_value:.6f} | "
                    f"Top5: {top5_value:.6f} | "
                    f"QDist: {q_score:.6f} | "
                    f"FullDiff: {full_loss:.6f} | "
                    f"Perp: {perplexity.item():.4f}"
                )

    mean_scores = np.array(mean_scores)
    max_scores = np.array(max_scores)
    top3_scores = np.array(top3_scores)
    top5_scores = np.array(top5_scores)
    top10_scores = np.array(top10_scores)

    q_scores = np.array(q_scores)
    full_losses = np.array(full_losses)
    perplexities = np.array(perplexities)

    print(f"\n==== {label_name} Summary ====")
    print("Num samples:", len(mean_scores))

    print("Spatial8 Mean score mean:", mean_scores.mean())
    print("Spatial8 Mean score std:", mean_scores.std())
    print("Spatial8 Mean score min:", mean_scores.min())
    print("Spatial8 Mean score max:", mean_scores.max())

    print("Spatial8 Max score mean:", max_scores.mean())
    print("Spatial8 Max score std:", max_scores.std())
    print("Spatial8 Max score min:", max_scores.min())
    print("Spatial8 Max score max:", max_scores.max())

    print("Spatial8 Top3 score mean:", top3_scores.mean())
    print("Spatial8 Top3 score std:", top3_scores.std())
    print("Spatial8 Top3 score min:", top3_scores.min())
    print("Spatial8 Top3 score max:", top3_scores.max())

    print("Spatial8 Top5 score mean:", top5_scores.mean())
    print("Spatial8 Top5 score std:", top5_scores.std())
    print("Spatial8 Top5 score min:", top5_scores.min())
    print("Spatial8 Top5 score max:", top5_scores.max())

    print("Spatial8 Top10 score mean:", top10_scores.mean())
    print("Spatial8 Top10 score std:", top10_scores.std())
    print("Spatial8 Top10 score min:", top10_scores.min())
    print("Spatial8 Top10 score max:", top10_scores.max())

    print("QDist mean:", q_scores.mean())
    print("QDist std:", q_scores.std())

    print("Full diff mean:", full_losses.mean())
    print("Full diff std:", full_losses.std())

    print("Perplexity mean:", perplexities.mean())
    print("Perplexity std:", perplexities.std())

    return {
        "mean": mean_scores,
        "max": max_scores,
        "top3": top3_scores,
        "top5": top5_scores,
        "top10": top10_scores,
        "qdist": q_scores,
        "fulldiff": full_losses,
        "perplexity": perplexities,
    }


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
        "spatial8_mean",
        "spatial8_max",
        "spatial8_top3",
        "spatial8_top5",
        "spatial8_top10",
        "qdistance",
        "full_diff",
        "perplexity"
    ])

    normal_result = evaluate_loader(
        normal_loader,
        "normal",
        normal_save_dir,
        writer
    )

    abnormal_result = evaluate_loader(
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

for key in ["mean", "max", "top3", "top5", "top10", "qdist", "fulldiff"]:

    normal_scores = normal_result[key]
    abnormal_scores = abnormal_result[key]

    print(f"\n[{key}]")
    print(
        f"Normal:   mean={normal_scores.mean():.6f}, "
        f"std={normal_scores.std():.6f}, "
        f"min={normal_scores.min():.6f}, "
        f"max={normal_scores.max():.6f}"
    )

    print(
        f"Abnormal: mean={abnormal_scores.mean():.6f}, "
        f"std={abnormal_scores.std():.6f}, "
        f"min={abnormal_scores.min():.6f}, "
        f"max={abnormal_scores.max():.6f}"
    )

    threshold = normal_scores.mean() + 3.0 * normal_scores.std()

    normal_fp = normal_scores > threshold
    abnormal_detected = abnormal_scores > threshold

    print(f"Threshold normal mean + 3std: {threshold:.6f}")
    print(
        "Normal false positive:",
        normal_fp.sum(),
        "/",
        len(normal_fp)
    )
    print(
        "Abnormal detected:",
        abnormal_detected.sum(),
        "/",
        len(abnormal_detected)
    )

print("\nCSV saved to:", csv_path)
print("Images saved to:", save_root)