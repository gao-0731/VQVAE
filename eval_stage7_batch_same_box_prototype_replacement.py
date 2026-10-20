import os
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
import pydicom
import numpy as np
from PIL import Image
import pandas as pd

from models.vqvae import VQVAE


# ============================================================
# Settings
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# abnormal data
abnormal_dir = "/app/data/temp/abnormal"

# normal train data
normal_train_dir = "/app/data/train"

# Stage1 full VQ-VAE
model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage1_full_final.pth"

save_root = "/app/results/eval_stage7_batch_same_box_prototype_replacement"
os.makedirs(save_root, exist_ok=True)

resize = 256

# normal prototype bank size
# 如果想用全部 normal train，可以设为 None
max_normal_samples = 500

# 每个 abnormal case 选 top-K prototype
top_k = 10

# 每张图可视化前几个 prototype
top_k_vis = 5

# replacement strength
alpha_list = [0.5, 0.7, 1.0]

# 用于保存图像的 alpha
vis_alpha = 1.0

# context ring padding in latent space
context_pad = 4

# 保存图片数量；None 表示全部保存
max_save_images = None


# ============================================================
# Manual latent boxes
#
# 512坐标:
# x=130~200, y=340~380
#
# 256坐标:
# x=65~100, y=170~190
#
# latent 64坐标:
# x=16~25, y=42~48
# ============================================================

manual_boxes = [
    {
        "name": "box_original",
        "ly1": 40,
        "ly2": 50,
        "lx1": 14,
        "lx2": 27,
    },
    {
        "name": "box_large",
        "ly1": 36,
        "ly2": 54,
        "lx1": 10,
        "lx2": 31,
    },
]


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
# Utilities
# ============================================================

def collect_dicom_paths(root_dir, max_samples=None):
    paths = []

    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".dcm"):
                paths.append(os.path.join(root, f))

    paths = sorted(paths)

    if max_samples is not None:
        paths = paths[:max_samples]

    return paths


def load_dicom_as_tensor(dicom_path):
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

    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)

    return image


def normalize_map(x):
    x = x.detach()
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def latent_to_image_box(ly1, ly2, lx1, lx2):
    scale = 4

    y1 = ly1 * scale
    y2 = ly2 * scale
    x1 = lx1 * scale
    x2 = lx2 * scale

    return y1, y2, x1, x2


def make_image_mask_from_latent_box(x, ly1, ly2, lx1, lx2):
    y1, y2, x1, x2 = latent_to_image_box(ly1, ly2, lx1, lx2)

    mask = torch.zeros_like(x)
    mask[:, :, y1:y2, x1:x2] = 1.0

    return mask, y1, y2, x1, x2


def make_overlay(x, mask):
    overlay = x.clone() * 0.75 + mask * 0.25
    overlay = torch.clamp(overlay, 0.0, 1.0)
    return overlay


def get_context_box(ly1, ly2, lx1, lx2, pad, latent_h=64, latent_w=64):
    cy1 = max(0, ly1 - pad)
    cy2 = min(latent_h, ly2 + pad)
    cx1 = max(0, lx1 - pad)
    cx2 = min(latent_w, lx2 + pad)

    return cy1, cy2, cx1, cx2


def make_context_ring_mask(
    cy1,
    cy2,
    cx1,
    cx2,
    ly1,
    ly2,
    lx1,
    lx2,
    device
):
    h = cy2 - cy1
    w = cx2 - cx1

    mask = torch.ones(1, 1, h, w, device=device)

    inner_y1 = ly1 - cy1
    inner_y2 = ly2 - cy1
    inner_x1 = lx1 - cx1
    inner_x2 = lx2 - cx1

    mask[:, :, inner_y1:inner_y2, inner_x1:inner_x2] = 0.0

    return mask


def compute_context_mse(
    target_zq,
    normal_zq,
    ly1,
    ly2,
    lx1,
    lx2,
    pad
):
    cy1, cy2, cx1, cx2 = get_context_box(
        ly1,
        ly2,
        lx1,
        lx2,
        pad
    )

    target_ctx = target_zq[:, :, cy1:cy2, cx1:cx2]
    normal_ctx = normal_zq[:, :, cy1:cy2, cx1:cx2]

    ring_mask = make_context_ring_mask(
        cy1,
        cy2,
        cx1,
        cx2,
        ly1,
        ly2,
        lx1,
        lx2,
        target_zq.device
    )

    diff = (target_ctx - normal_ctx) ** 2
    diff = diff * ring_mask

    mse = diff.sum() / (ring_mask.sum() * target_zq.shape[1] + 1e-8)

    return mse.item()


def replace_with_prototype(
    z_q,
    proto_zq,
    ly1,
    ly2,
    lx1,
    lx2,
    alpha
):
    z_corr = z_q.clone()

    z_corr[:, :, ly1:ly2, lx1:lx2] = (
        (1.0 - alpha) * z_q[:, :, ly1:ly2, lx1:lx2]
        + alpha * proto_zq[:, :, ly1:ly2, lx1:lx2]
    )

    return z_corr


def compute_scores(x, recon, region_mask):
    diff = torch.abs(x - recon)

    whole_diff = diff.mean().item()

    region_diff = (
        diff * region_mask
    ).sum().item() / (region_mask.sum().item() + 1e-8)

    outside_mask = 1.0 - region_mask
    outside_diff = (
        diff * outside_mask
    ).sum().item() / (outside_mask.sum().item() + 1e-8)

    return diff, whole_diff, region_diff, outside_diff


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

checkpoint = torch.load(model_path, map_location=device)

if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

model.eval()

print("Loaded model:", model_path)


# ============================================================
# Model forward helper
# ============================================================

def extract_zq_recon_qdist(model, x):
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

    return recon, z_q, q_distance_map, perplexity


# ============================================================
# Build normal z_q bank
# ============================================================

def build_normal_zq_bank(model, normal_paths):
    bank = []

    print()
    print("Building normal z_q bank...")
    print("Normal samples:", len(normal_paths))

    with torch.no_grad():
        for i, path in enumerate(normal_paths):
            try:
                x_n = load_dicom_as_tensor(path).to(device)

                _, zq_n, _, _ = extract_zq_recon_qdist(
                    model,
                    x_n
                )

                bank.append({
                    "path": path,
                    "case_name": os.path.basename(path).replace(".dcm", ""),
                    "zq": zq_n.detach().cpu(),
                })

                if (i + 1) % 20 == 0:
                    print(f"Processed normal {i+1}/{len(normal_paths)}")

            except Exception as e:
                print("Skip error:", path)
                print(e)

    print("Bank size:", len(bank))

    return bank


normal_paths = collect_dicom_paths(
    normal_train_dir,
    max_samples=max_normal_samples
)

normal_bank = build_normal_zq_bank(
    model,
    normal_paths
)


# ============================================================
# Evaluate one abnormal case
# ============================================================

def evaluate_case(case_idx, target_dicom_path):
    target_case_name = os.path.basename(target_dicom_path).replace(".dcm", "")

    case_save_dir = os.path.join(save_root, target_case_name)
    os.makedirs(case_save_dir, exist_ok=True)

    target_x = load_dicom_as_tensor(target_dicom_path).to(device)

    with torch.no_grad():
        recon_original, target_zq, q_distance_map, perplexity = extract_zq_recon_qdist(
            model,
            target_x
        )

        diff_original = torch.abs(target_x - recon_original)

        q_map = q_distance_map.unsqueeze(1)
        q_map = F.interpolate(
            q_map,
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

    case_records = []

    print()
    print("=" * 80)
    print(f"Case {case_idx}: {target_case_name}")
    print("Path:", target_dicom_path)
    print("ordinary_diff:", diff_original.mean().item())
    print("perplexity:", float(perplexity.item()))

    for box in manual_boxes:
        box_name = box["name"]

        ly1 = box["ly1"]
        ly2 = box["ly2"]
        lx1 = box["lx1"]
        lx2 = box["lx2"]

        lesion_mask, y1_img, y2_img, x1_img, x2_img = make_image_mask_from_latent_box(
            target_x,
            ly1,
            ly2,
            lx1,
            lx2
        )

        overlay = make_overlay(target_x, lesion_mask)

        ordinary_whole_diff = diff_original.mean().item()

        ordinary_region_diff = (
            diff_original * lesion_mask
        ).sum().item() / (lesion_mask.sum().item() + 1e-8)

        ordinary_outside_diff = (
            diff_original * (1.0 - lesion_mask)
        ).sum().item() / ((1.0 - lesion_mask).sum().item() + 1e-8)

        print()
        print("Box:", box_name)
        print("latent y:", ly1, ly2, "latent x:", lx1, lx2)
        print("image  y:", y1_img, y2_img, "image  x:", x1_img, x2_img)
        print(
            f"ordinary: whole={ordinary_whole_diff:.6f}, "
            f"region={ordinary_region_diff:.6f}, "
            f"outside={ordinary_outside_diff:.6f}"
        )

        # ----------------------------------------------------
        # Search top-K normal prototypes by context similarity
        # ----------------------------------------------------
        candidate_list = []

        target_zq_device = target_zq.detach()

        for item in normal_bank:
            normal_zq = item["zq"].to(device)

            context_mse = compute_context_mse(
                target_zq_device,
                normal_zq,
                ly1,
                ly2,
                lx1,
                lx2,
                context_pad
            )

            candidate_list.append({
                "path": item["path"],
                "case_name": item["case_name"],
                "zq": item["zq"],
                "context_mse": context_mse,
            })

        candidate_list = sorted(
            candidate_list,
            key=lambda x: x["context_mse"]
        )

        selected = candidate_list[:top_k]

        print("Top prototypes:")
        for r, c in enumerate(selected[:5]):
            print(
                f"rank={r+1}, mse={c['context_mse']:.6f}, "
                f"case={c['case_name']}"
            )

        # ----------------------------------------------------
        # Image grid
        # ----------------------------------------------------
        images = [
            target_x.cpu(),                         # 1 input
            recon_original.cpu(),                   # 2 ordinary recon
            normalize_map(diff_original).cpu(),     # 3 ordinary diff
            overlay.cpu(),                          # 4 input + mask overlay
            lesion_mask.cpu(),                      # 5 binary mask
            normalize_map(q_map).cpu(),             # 6 qdistance
        ]

        # ----------------------------------------------------
        # Evaluate prototypes
        # ----------------------------------------------------
        for rank, c in enumerate(selected):
            proto_zq = c["zq"].to(device)

            for alpha in alpha_list:
                with torch.no_grad():
                    z_corr = replace_with_prototype(
                        target_zq,
                        proto_zq,
                        ly1,
                        ly2,
                        lx1,
                        lx2,
                        alpha
                    )

                    recon_corr = model.decoder(z_corr)

                    diff_corr, whole_diff, region_diff, outside_diff = compute_scores(
                        target_x,
                        recon_corr,
                        lesion_mask
                    )

                case_records.append({
                    "target_case": target_case_name,
                    "target_path": target_dicom_path,
                    "box_name": box_name,

                    "latent_y1": ly1,
                    "latent_y2": ly2,
                    "latent_x1": lx1,
                    "latent_x2": lx2,

                    "image_y1": y1_img,
                    "image_y2": y2_img,
                    "image_x1": x1_img,
                    "image_x2": x2_img,

                    "prototype_rank": rank + 1,
                    "prototype_case": c["case_name"],
                    "prototype_path": c["path"],
                    "context_mse": c["context_mse"],

                    "alpha": alpha,

                    "whole_diff": whole_diff,
                    "region_diff": region_diff,
                    "outside_diff": outside_diff,

                    "ordinary_whole_diff": ordinary_whole_diff,
                    "ordinary_region_diff": ordinary_region_diff,
                    "ordinary_outside_diff": ordinary_outside_diff,

                    "region_gain_vs_ordinary": region_diff - ordinary_region_diff,
                    "outside_gain_vs_ordinary": outside_diff - ordinary_outside_diff,

                    "gain_ratio": (
                        (region_diff - ordinary_region_diff)
                        / (outside_diff - ordinary_outside_diff + 1e-8)
                    ),
                    "perplexity": float(perplexity.item()),
                })

            # visualization only
            if rank < top_k_vis:
                proto_zq = c["zq"].to(device)

                with torch.no_grad():
                    z_corr = replace_with_prototype(
                        target_zq,
                        proto_zq,
                        ly1,
                        ly2,
                        lx1,
                        lx2,
                        vis_alpha
                    )

                    recon_corr = model.decoder(z_corr)
                    diff_corr = torch.abs(target_x - recon_corr)

                images.extend([
                    recon_corr.cpu(),
                    normalize_map(diff_corr).cpu(),
                ])

        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------
        should_save = True
        if max_save_images is not None and case_idx >= max_save_images:
            should_save = False

        if should_save:
            grid = vutils.make_grid(
                torch.cat(images, dim=0),
                nrow=6 + 2 * min(top_k_vis, len(selected)),
                normalize=True
            )

            save_img_path = os.path.join(
                case_save_dir,
                f"{target_case_name}_{box_name}_prototype_replacement.png"
            )

            vutils.save_image(grid, save_img_path)

            print("Saved image:", save_img_path)

    return case_records


# ============================================================
# Run batch
# ============================================================

abnormal_paths = collect_dicom_paths(abnormal_dir)

print()
print("Abnormal cases:", len(abnormal_paths))

all_records = []

for idx, path in enumerate(abnormal_paths):
    records = evaluate_case(idx, path)
    all_records.extend(records)

# ============================================================
# Save all CSV
# ============================================================

df = pd.DataFrame(all_records)

csv_path = os.path.join(
    save_root,
    "stage7_batch_same_box_prototype_replacement_scores.csv"
)

df.to_csv(csv_path, index=False)

print()
print("Saved CSV:", csv_path)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 80)
print("Summary by box / alpha / prototype rank <= 10")

summary = (
    df
    .groupby(["box_name", "alpha"])
    .agg(
        region_gain_mean=("region_gain_vs_ordinary", "mean"),
        region_gain_std=("region_gain_vs_ordinary", "std"),
        outside_gain_mean=("outside_gain_vs_ordinary", "mean"),
        outside_gain_std=("outside_gain_vs_ordinary", "std"),
        gain_ratio_mean=("gain_ratio", "mean"),
        region_diff_mean=("region_diff", "mean"),
        outside_diff_mean=("outside_diff", "mean"),
        n=("target_case", "count"),
    )
    .reset_index()
)

print(summary.to_string(index=False))

summary_csv_path = os.path.join(
    save_root,
    "stage7_batch_summary_by_box_alpha.csv"
)

summary.to_csv(summary_csv_path, index=False)

print()
print("Saved summary CSV:", summary_csv_path)

print()
print("Image columns:")
print("1 input")
print("2 ordinary recon")
print("3 ordinary diff")
print("4 input + mask overlay")
print("5 binary mask")
print("6 qdistance")
print("Then for top prototypes:")
print("prototype1 recon")
print("prototype1 diff")
print("prototype2 recon")
print("prototype2 diff")
print("...")
print(f"Visualization alpha = {vis_alpha}")