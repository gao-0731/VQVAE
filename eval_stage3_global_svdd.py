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
# 必须和训练一致
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

normal_data_dir = "/app/data/temp/normal"
abnormal_data_dir = "/app/data/temp/abnormal"

# Stage3 模型
model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage3_global_svdd_final.pth"

# 如果 final 效果不好，可以换 best
# model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage3_global_svdd_best.pth"

# 训练 SVDD 时用的 center
center_path = "/app/results/svdd_center_global_stage2_v2.pth"
# center_path = "/app/results/svdd_center_global_stage2_v2_train.pth"

save_root = "/app/results/eval_stage3_global_svdd"
os.makedirs(save_root, exist_ok=True)

normal_save_dir = os.path.join(save_root, "normal")
abnormal_save_dir = os.path.join(save_root, "abnormal")

os.makedirs(normal_save_dir, exist_ok=True)
os.makedirs(abnormal_save_dir, exist_ok=True)

csv_path = os.path.join(save_root, "svdd_scores.csv")


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
        # 必须和训练一致
        # =============================

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))

        image = image * slope + intercept

        # =============================
        # Lung window
        # 必须和 Stage1 / Stage2 / Stage3 一致
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
# Load Center
# =============================

center_ckpt = torch.load(
    center_path,
    map_location=device
)

center = center_ckpt["center"].to(device)

print("Loaded center:", center_path)
print("Center shape:", center.shape)
print("Center mean:", center.mean().item())
print("Center std:", center.std().item())


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
        z_q
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
# Normalize for visualization
# =============================

def normalize_map(x):
    """
    x: tensor [1,1,H,W] or [1,H,W]
    """

    x = x.detach()

    x_min = x.min()
    x_max = x.max()

    return (x - x_min) / (x_max - x_min + 1e-8)


def save_eval_image(
    x,
    x_hat,
    full_diff,
    q_distance_map,
    save_path
):
    """
    保存一张横向比较图：
    input | recon | full_diff | qdistance
    """

    # x, x_hat, full_diff: [1,1,256,256]
    # q_distance_map: [1,64,64]

    q_map = q_distance_map.unsqueeze(1)  # [1,1,64,64]

    q_map = F.interpolate(
        q_map,
        size=x.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    q_map = normalize_map(q_map)
    diff_vis = normalize_map(full_diff)

    grid = vutils.make_grid(
        torch.cat([
            x.cpu(),
            x_hat.cpu(),
            diff_vis.cpu(),
            q_map.cpu()
        ], dim=0),
        nrow=4,
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

    scores = []
    full_losses = []
    perplexities = []

    with torch.no_grad():

        for idx, batch in enumerate(loader):

            x, paths = batch
            x = x.to(device)

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

            full_diff = torch.abs(x - x_hat)

            full_loss = torch.mean(full_diff).item()

            # =============================
            # Global SVDD score
            # =============================

            z_global = z_q.mean(
                dim=[2, 3]
            )  # [B,64]

            svdd_score = torch.sum(
                (z_global - center) ** 2,
                dim=1
            )  # [B]

            svdd_score_value = svdd_score.item()

            scores.append(svdd_score_value)
            full_losses.append(full_loss)
            perplexities.append(perplexity.item())

            dicom_path = paths[0]
            case_name = os.path.basename(dicom_path)

            csv_writer.writerow([
                label_name,
                idx,
                case_name,
                dicom_path,
                svdd_score_value,
                full_loss,
                perplexity.item()
            ])

            # =============================
            # Save images
            # =============================

            if idx < 200:
                image_save_path = os.path.join(
                    save_dir,
                    f"{idx:04d}_{case_name}.png"
                )

                save_eval_image(
                    x,
                    x_hat,
                    full_diff,
                    q_distance_map,
                    image_save_path
                )

            if idx % 50 == 0:
                print(
                    f"[{label_name}] "
                    f"{idx}/{len(loader)} | "
                    f"SVDD: {svdd_score_value:.6f} | "
                    f"FullDiff: {full_loss:.6f} | "
                    f"Perplexity: {perplexity.item():.4f}"
                )

    scores = np.array(scores)
    full_losses = np.array(full_losses)
    perplexities = np.array(perplexities)

    print(f"\n==== {label_name} Summary ====")
    print("Num samples:", len(scores))
    print("SVDD score mean:", scores.mean())
    print("SVDD score std:", scores.std())
    print("SVDD score min:", scores.min())
    print("SVDD score max:", scores.max())
    print("Full diff mean:", full_losses.mean())
    print("Full diff std:", full_losses.std())
    print("Perplexity mean:", perplexities.mean())
    print("Perplexity std:", perplexities.std())

    return scores, full_losses, perplexities


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
        "svdd_score",
        "full_diff",
        "perplexity"
    ])

    normal_scores, normal_diffs, normal_perps = evaluate_loader(
        normal_loader,
        "normal",
        normal_save_dir,
        writer
    )

    abnormal_scores, abnormal_diffs, abnormal_perps = evaluate_loader(
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
    f"Normal SVDD:   mean={normal_scores.mean():.6f}, "
    f"std={normal_scores.std():.6f}, "
    f"min={normal_scores.min():.6f}, "
    f"max={normal_scores.max():.6f}"
)

print(
    f"Abnormal SVDD: mean={abnormal_scores.mean():.6f}, "
    f"std={abnormal_scores.std():.6f}, "
    f"min={abnormal_scores.min():.6f}, "
    f"max={abnormal_scores.max():.6f}"
)

print(
    f"Normal FullDiff:   mean={normal_diffs.mean():.6f}, "
    f"std={normal_diffs.std():.6f}"
)

print(
    f"Abnormal FullDiff: mean={abnormal_diffs.mean():.6f}, "
    f"std={abnormal_diffs.std():.6f}"
)

print("CSV saved to:", csv_path)
print("Images saved to:", save_root)


# =============================
# Optional: simple threshold check
# =============================

threshold = normal_scores.mean() + 3.0 * normal_scores.std()

abnormal_detected = abnormal_scores > threshold
normal_false_positive = normal_scores > threshold

print("\n==== Simple Threshold: normal mean + 3std ====")
print("Threshold:", threshold)
print(
    "Normal false positive:",
    normal_false_positive.sum(),
    "/",
    len(normal_false_positive)
)
print(
    "Abnormal detected:",
    abnormal_detected.sum(),
    "/",
    len(abnormal_detected)
)