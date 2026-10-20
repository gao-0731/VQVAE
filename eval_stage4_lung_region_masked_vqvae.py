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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

resize = 256
batch_size = 1

torch.manual_seed(42)
np.random.seed(42)


# =============================
# Model Parameters
# 必须和 Stage4 训练一致
# =============================

n_hiddens = 128
n_residual_hiddens = 64
n_residual_layers = 1

embedding_dim = 64
n_embeddings = 64
beta = 0.25


# =============================
# Mask Settings
# 必须和 Stage4 训练一致
# =============================

mask_ratio = 0.05
block_size = 4
lung_dilate_kernel = 21


# =============================
# Paths
# =============================

normal_data_dir = "/app/data/temp/normal"
abnormal_data_dir = "/app/data/temp/abnormal"

# Stage4 model
model_path = "/app/results/vqvae_data_vqvae_ct_lung_region_masked_stage4_final.pth"

# 如果 final 重建不好，可以换 best
# model_path = "/app/results/vqvae_data_vqvae_ct_lung_region_masked_stage4_best.pth"

save_root = "/app/results/eval_stage4_lung_region_masked_vqvae"
os.makedirs(save_root, exist_ok=True)

normal_save_dir = os.path.join(save_root, "normal")
abnormal_save_dir = os.path.join(save_root, "abnormal")

os.makedirs(normal_save_dir, exist_ok=True)
os.makedirs(abnormal_save_dir, exist_ok=True)

csv_path = os.path.join(save_root, "stage4_eval_scores.csv")


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

        # Lung window [-1000, 400] -> [0, 1]
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

print("Loaded Stage4 model:", model_path)


# =============================
# Lung-region mask helpers
# =============================

def make_lung_region_mask(x):
    """
    x: [B,1,256,256], normalized lung window [0,1]

    return:
        lung_region: [B,1,256,256]
    """

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

    latent_mask = latent_mask * lung_latent

    image_mask = F.interpolate(
        latent_mask,
        size=x.shape[-2:],
        mode="nearest"
    )

    return latent_mask, image_mask, lung_region


# =============================
# Forward functions
# =============================

def forward_no_mask(model, x):
    """
    普通完整输入重建
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


def forward_lung_masked(model, x):
    """
    lung region + boundary 内 latent mask 后重建
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

    latent_h, latent_w = z_e.shape[-2:]

    latent_mask, image_mask, lung_region = make_lung_latent_block_mask(
        x,
        latent_h=latent_h,
        latent_w=latent_w,
        mask_ratio=mask_ratio,
        block_size=block_size
    )

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
        x_hat,
        q_distance_map,
        z_q,
        perplexity,
        image_mask,
        lung_region
    )


# =============================
# Visualization helpers
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
    recon_no_mask,
    diff_no_mask,
    recon_lung_masked,
    diff_lung_masked,
    q_distance_map,
    image_mask,
    lung_region,
    save_path
):
    """
    保存：
    input | recon_no_mask | diff_no_mask | recon_lung_masked |
    diff_lung_masked | qdistance | mask | lung_region
    """

    q_map = q_distance_map.unsqueeze(1)
    q_map = upsample_map(q_map, x.shape[-2:])

    diff_no_mask_vis = percentile_clip_normalize(diff_no_mask)
    diff_lung_masked_vis = percentile_clip_normalize(diff_lung_masked)
    q_vis = percentile_clip_normalize(q_map)

    grid = vutils.make_grid(
        torch.cat([
            x.cpu(),
            recon_no_mask.cpu(),
            diff_no_mask_vis.cpu(),
            recon_lung_masked.cpu(),
            diff_lung_masked_vis.cpu(),
            q_vis.cpu(),
            image_mask.cpu(),
            lung_region.cpu(),
        ], dim=0),
        nrow=8,
        normalize=True
    )

    vutils.save_image(
        grid,
        save_path
    )


# =============================
# Eval function
# =============================

def evaluate_loader(
    loader,
    label_name,
    save_dir,
    csv_writer
):

    no_mask_diffs = []
    lung_masked_diffs = []
    masked_region_diffs = []
    q_scores = []
    perplexities_no_mask = []
    perplexities_lung_masked = []

    with torch.no_grad():

        for idx, batch in enumerate(loader):

            x, paths = batch
            x = x.to(device)

            # 普通 no-mask reconstruction
            (
                recon_no_mask,
                q_distance_map_no_mask,
                z_q_no_mask,
                perplexity_no_mask
            ) = forward_no_mask(
                model,
                x
            )

            # lung-region masked reconstruction
            (
                recon_lung_masked,
                q_distance_map_lung_masked,
                z_q_lung_masked,
                perplexity_lung_masked,
                image_mask,
                lung_region
            ) = forward_lung_masked(
                model,
                x
            )

            diff_no_mask = torch.abs(
                x - recon_no_mask
            )

            diff_lung_masked = torch.abs(
                x - recon_lung_masked
            )

            no_mask_diff_score = diff_no_mask.mean().item()
            lung_masked_diff_score = diff_lung_masked.mean().item()

            masked_region_diff_score = (
                (diff_lung_masked * image_mask).sum()
                / (image_mask.sum() + 1e-8)
            ).item()

            q_score = q_distance_map_no_mask.mean().item()

            no_mask_diffs.append(no_mask_diff_score)
            lung_masked_diffs.append(lung_masked_diff_score)
            masked_region_diffs.append(masked_region_diff_score)
            q_scores.append(q_score)
            perplexities_no_mask.append(perplexity_no_mask.item())
            perplexities_lung_masked.append(perplexity_lung_masked.item())

            dicom_path = paths[0]
            case_name = os.path.basename(dicom_path)

            csv_writer.writerow([
                label_name,
                idx,
                case_name,
                dicom_path,
                no_mask_diff_score,
                lung_masked_diff_score,
                masked_region_diff_score,
                q_score,
                perplexity_no_mask.item(),
                perplexity_lung_masked.item()
            ])

            if idx < 300:
                image_save_path = os.path.join(
                    save_dir,
                    f"{idx:04d}_{case_name}.png"
                )

                save_eval_image(
                    x,
                    recon_no_mask,
                    diff_no_mask,
                    recon_lung_masked,
                    diff_lung_masked,
                    q_distance_map_no_mask,
                    image_mask,
                    lung_region,
                    image_save_path
                )

            if idx % 50 == 0:
                print(
                    f"[{label_name}] "
                    f"{idx}/{len(loader)} | "
                    f"NoMaskDiff: {no_mask_diff_score:.6f} | "
                    f"LungMaskedDiff: {lung_masked_diff_score:.6f} | "
                    f"MaskedRegionDiff: {masked_region_diff_score:.6f} | "
                    f"QDist: {q_score:.6f} | "
                    f"PerpNoMask: {perplexity_no_mask.item():.4f} | "
                    f"PerpMasked: {perplexity_lung_masked.item():.4f}"
                )

    no_mask_diffs = np.array(no_mask_diffs)
    lung_masked_diffs = np.array(lung_masked_diffs)
    masked_region_diffs = np.array(masked_region_diffs)
    q_scores = np.array(q_scores)
    perplexities_no_mask = np.array(perplexities_no_mask)
    perplexities_lung_masked = np.array(perplexities_lung_masked)

    print(f"\n==== {label_name} Summary ====")
    print("Num samples:", len(no_mask_diffs))

    print("NoMaskDiff mean:", no_mask_diffs.mean())
    print("NoMaskDiff std:", no_mask_diffs.std())
    print("NoMaskDiff min:", no_mask_diffs.min())
    print("NoMaskDiff max:", no_mask_diffs.max())

    print("LungMaskedDiff mean:", lung_masked_diffs.mean())
    print("LungMaskedDiff std:", lung_masked_diffs.std())
    print("LungMaskedDiff min:", lung_masked_diffs.min())
    print("LungMaskedDiff max:", lung_masked_diffs.max())

    print("MaskedRegionDiff mean:", masked_region_diffs.mean())
    print("MaskedRegionDiff std:", masked_region_diffs.std())
    print("MaskedRegionDiff min:", masked_region_diffs.min())
    print("MaskedRegionDiff max:", masked_region_diffs.max())

    print("QDist mean:", q_scores.mean())
    print("QDist std:", q_scores.std())

    print("Perplexity no mask mean:", perplexities_no_mask.mean())
    print("Perplexity no mask std:", perplexities_no_mask.std())

    print("Perplexity lung masked mean:", perplexities_lung_masked.mean())
    print("Perplexity lung masked std:", perplexities_lung_masked.std())

    return {
        "no_mask_diff": no_mask_diffs,
        "lung_masked_diff": lung_masked_diffs,
        "masked_region_diff": masked_region_diffs,
        "qdist": q_scores,
        "perplexity_no_mask": perplexities_no_mask,
        "perplexity_lung_masked": perplexities_lung_masked,
    }


# =============================
# Run eval
# =============================

with open(csv_path, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "label",
        "index",
        "filename",
        "path",
        "no_mask_diff",
        "lung_masked_diff",
        "masked_region_diff",
        "qdistance",
        "perplexity_no_mask",
        "perplexity_lung_masked"
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
# Final comparison
# =============================

print("\n==============================")
print("Final Comparison")
print("==============================")

for key in [
    "no_mask_diff",
    "lung_masked_diff",
    "masked_region_diff",
    "qdist",
    "perplexity_no_mask",
    "perplexity_lung_masked"
]:

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