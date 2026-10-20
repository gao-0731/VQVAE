import os
import math
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import pandas as pd
import pydicom
import numpy as np
from PIL import Image

from models.vqvae import VQVAE


# ============================================================
# Settings
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

normal_data_dir = "/app/data/temp/normal"
abnormal_data_dir = "/app/data/temp/abnormal"

model_path = "/app/results/vqvae_ct_lungwindow_stage10_normal_inpainting_best.pth"

save_root = "/app/results/eval_stage10_sliding_mask_detection"
os.makedirs(save_root, exist_ok=True)

resize = 256

# ------------------------------------------------------------
# Sliding mask settings
# ------------------------------------------------------------
mask_size_list = [24, 32, 40]
stride = 16
top_n = 3
nms_iou_thr = 0.30

fill_mode = "local_mean"   # "local_mean" or "constant"
fill_value = 0.08
local_mean_pad = 8

min_lung_ratio = 0.25

# score
outside_weight = 0.30

# region score mode
score_mode = "topk"    # "topk" or "mean"
topk_ratio = 0.10

# limit optional
max_cases_per_set = None   # None means all


# ============================================================
# VQ-VAE Parameters
# 必须和训练一致
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

def collect_dicom_paths(root_dir):
    paths = []

    if not os.path.exists(root_dir):
        print("Warning: dir not found:", root_dir)
        return paths

    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".dcm"):
                paths.append(os.path.join(root, f))

    paths = sorted(paths)

    if max_cases_per_set is not None:
        paths = paths[:max_cases_per_set]

    return paths


def load_dicom_lung_window(dicom_path):
    ds = pydicom.dcmread(dicom_path)

    image = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    image = image * slope + intercept

    # lung window
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


def make_lung_mask(x):
    """
    x: [1,1,H,W]
    """

    lung = ((x > 0.02) & (x < 0.45)).float()

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
    x: [1,1,H,W]
    """

    _, _, h, w = x.shape

    yy1 = max(0, y1 - pad)
    yy2 = min(h, y2 + pad)
    xx1 = max(0, x1 - pad)
    xx2 = min(w, x2 + pad)

    region = x[:, :, yy1:yy2, xx1:xx2].clone()

    inner_y1 = y1 - yy1
    inner_y2 = y2 - yy1
    inner_x1 = x1 - xx1
    inner_x2 = x2 - xx1

    valid = torch.ones_like(region)
    valid[:, :, inner_y1:inner_y2, inner_x1:inner_x2] = 0.0

    if valid.sum() < 1:
        return torch.tensor(fill_value, dtype=x.dtype, device=x.device)

    mean_value = (region * valid).sum() / (valid.sum() + 1e-8)

    return mean_value


def apply_block_mask(x, y1, y2, x1, x2):
    """
    x: [1,1,H,W]
    """
    x_masked = x.clone()
    mask = torch.zeros_like(x)

    if fill_mode == "local_mean":
        fill = get_local_mean_fill(
            x,
            y1,
            y2,
            x1,
            x2,
            pad=local_mean_pad
        )
    elif fill_mode == "constant":
        fill = torch.tensor(
            fill_value,
            dtype=x.dtype,
            device=x.device
        )
    else:
        raise ValueError(f"Unknown fill_mode: {fill_mode}")

    x_masked[:, :, y1:y2, x1:x2] = fill
    mask[:, :, y1:y2, x1:x2] = 1.0

    return x_masked, mask


def make_overlay(x, mask):
    overlay = torch.clamp(x * 0.75 + mask * 0.25, 0.0, 1.0)
    return overlay


def compute_masked_region_score(diff, mask, mode="topk", topk_ratio=0.10):
    """
    diff, mask: [1,1,H,W]
    return scalar
    """
    vals = diff[mask > 0.5]

    if vals.numel() == 0:
        return 0.0

    if mode == "mean":
        return vals.mean().item()

    elif mode == "topk":
        k = max(1, int(math.ceil(vals.numel() * topk_ratio)))
        topv, _ = torch.topk(vals, k)
        return topv.mean().item()

    else:
        raise ValueError(f"Unknown score mode: {mode}")


def compute_mean_outside_score(diff, mask):
    outside = diff[mask < 0.5]

    if outside.numel() == 0:
        return 0.0

    return outside.mean().item()


def iou(box_a, box_b):
    ay1, ay2, ax1, ax2 = box_a
    by1, by2, bx1, bx2 = box_b

    inter_y1 = max(ay1, by1)
    inter_y2 = min(ay2, by2)
    inter_x1 = max(ax1, bx1)
    inter_x2 = min(ax2, bx2)

    inter_h = max(0, inter_y2 - inter_y1)
    inter_w = max(0, inter_x2 - inter_x1)
    inter = inter_h * inter_w

    area_a = (ay2 - ay1) * (ax2 - ax1)
    area_b = (by2 - by1) * (bx2 - bx1)

    union = area_a + area_b - inter + 1e-8

    return inter / union


def nms_candidates(candidates, iou_thr=0.30, top_n=3):
    """
    candidates: list of dict, each contains "box" and "score"
    """
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    selected = []

    for cand in candidates:
        keep = True
        for sel in selected:
            if iou(cand["box"], sel["box"]) > iou_thr:
                keep = False
                break
        if keep:
            selected.append(cand)
        if len(selected) >= top_n:
            break

    return selected


# ============================================================
# Load model
# ============================================================

def load_model(model_path):
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
    return model


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

    return recon, perplexity, q_distance_map


model = load_model(model_path)


# ============================================================
# Candidate generation
# ============================================================

def generate_candidates_for_one_image(model, x):
    """
    x: [1,1,256,256]
    return:
        ordinary_recon
        ordinary_diff
        selected_candidates
        all_candidates
    """

    with torch.no_grad():
        ordinary_recon, ordinary_perplexity, ordinary_qmap = forward_vqvae(model, x)

    ordinary_diff = torch.abs(x - ordinary_recon)

    lung_mask = make_lung_mask(x)

    _, _, h, w = x.shape

    all_candidates = []

    for block_size in mask_size_list:
        for y1 in range(0, h - block_size + 1, stride):
            for x1 in range(0, w - block_size + 1, stride):
                y2 = y1 + block_size
                x2 = x1 + block_size

                lung_ratio = lung_mask[:, :, y1:y2, x1:x2].mean().item()

                if lung_ratio < min_lung_ratio:
                    continue

                x_masked, mask = apply_block_mask(x, y1, y2, x1, x2)

                with torch.no_grad():
                    recon, perplexity, qmap = forward_vqvae(model, x_masked)

                diff = torch.abs(x - recon)

                ordinary_region_score = compute_masked_region_score(
                    ordinary_diff,
                    mask,
                    mode=score_mode,
                    topk_ratio=topk_ratio
                )

                masked_region_score = compute_masked_region_score(
                    diff,
                    mask,
                    mode=score_mode,
                    topk_ratio=topk_ratio
                )

                ordinary_outside_score = compute_mean_outside_score(
                    ordinary_diff,
                    mask
                )

                masked_outside_score = compute_mean_outside_score(
                    diff,
                    mask
                )

                region_gain = masked_region_score - ordinary_region_score
                outside_gain = masked_outside_score - ordinary_outside_score

                score = region_gain - outside_weight * max(0.0, outside_gain)

                all_candidates.append({
                    "box": (y1, y2, x1, x2),
                    "mask_size": block_size,
                    "lung_ratio": lung_ratio,

                    "ordinary_region_score": ordinary_region_score,
                    "masked_region_score": masked_region_score,

                    "ordinary_outside_score": ordinary_outside_score,
                    "masked_outside_score": masked_outside_score,

                    "region_gain": region_gain,
                    "outside_gain": outside_gain,
                    "score": score,
                })

    selected_candidates = nms_candidates(
        all_candidates,
        iou_thr=nms_iou_thr,
        top_n=top_n
    )

    return ordinary_recon, ordinary_diff, ordinary_perplexity, selected_candidates, all_candidates


# ============================================================
# Visualization and evaluation
# ============================================================

def save_case_visualization(
    x,
    ordinary_recon,
    ordinary_diff,
    selected_candidates,
    case_save_dir,
    case_name
):
    grid_items = [
        x.cpu(),
        ordinary_recon.cpu(),
        normalize_map(ordinary_diff).cpu(),
    ]

    candidate_records = []

    for rank, cand in enumerate(selected_candidates, start=1):
        y1, y2, x1, x2 = cand["box"]

        x_masked, mask = apply_block_mask(x, y1, y2, x1, x2)

        with torch.no_grad():
            recon, perplexity, qmap = forward_vqvae(model, x_masked)

        diff = torch.abs(x - recon)

        overlay = make_overlay(x, mask)

        grid_items.extend([
            overlay.cpu(),
            x_masked.cpu(),
            recon.cpu(),
            normalize_map(diff).cpu(),
        ])

        candidate_records.append({
            "rank": rank,
            "mask_size": cand["mask_size"],
            "y1": y1,
            "y2": y2,
            "x1": x1,
            "x2": x2,
            "lung_ratio": cand["lung_ratio"],
            "ordinary_region_score": cand["ordinary_region_score"],
            "masked_region_score": cand["masked_region_score"],
            "ordinary_outside_score": cand["ordinary_outside_score"],
            "masked_outside_score": cand["masked_outside_score"],
            "region_gain": cand["region_gain"],
            "outside_gain": cand["outside_gain"],
            "score": cand["score"],
            "perplexity": float(perplexity.item()),
        })

    ncol = 3 + 4 * len(selected_candidates)

    grid = vutils.make_grid(
        torch.cat(grid_items, dim=0),
        nrow=ncol,
        normalize=True
    )

    save_img_path = os.path.join(
        case_save_dir,
        f"{case_name}_top_candidates.png"
    )

    vutils.save_image(grid, save_img_path)

    return save_img_path, candidate_records


def evaluate_one_set(set_name, data_dir):
    print()
    print("=" * 80)
    print("Evaluating set:", set_name)
    print("Dir:", data_dir)
    print("=" * 80)

    paths = collect_dicom_paths(data_dir)
    print(f"{set_name} cases:", len(paths))

    set_save_dir = os.path.join(save_root, set_name)
    os.makedirs(set_save_dir, exist_ok=True)

    all_rows = []

    for idx, dicom_path in enumerate(paths):
        case_name = os.path.basename(dicom_path).replace(".dcm", "")
        case_save_dir = os.path.join(set_save_dir, case_name)
        os.makedirs(case_save_dir, exist_ok=True)

        print(f"[{set_name}] {idx+1}/{len(paths)}  {case_name}")

        x = load_dicom_lung_window(dicom_path).to(device)

        ordinary_recon, ordinary_diff, ordinary_perplexity, selected_candidates, all_candidates = \
            generate_candidates_for_one_image(model, x)

        save_img_path, candidate_records = save_case_visualization(
            x,
            ordinary_recon,
            ordinary_diff,
            selected_candidates,
            case_save_dir,
            case_name
        )

        for row in candidate_records:
            row["set_name"] = set_name
            row["case_name"] = case_name
            row["path"] = dicom_path
            row["ordinary_whole_diff"] = ordinary_diff.mean().item()
            row["ordinary_perplexity"] = float(ordinary_perplexity.item())
            row["image_path"] = save_img_path
            all_rows.append(row)

        print("  saved:", save_img_path)

        if len(candidate_records) > 0:
            best = candidate_records[0]
            print(
                f"  top1 score={best['score']:.6f}, "
                f"region_gain={best['region_gain']:.6f}, "
                f"outside_gain={best['outside_gain']:.6f}, "
                f"mask={best['mask_size']}, "
                f"box=({best['y1']}:{best['y2']}, {best['x1']}:{best['x2']})"
            )

    return all_rows


# ============================================================
# Run
# ============================================================

all_rows = []

all_rows.extend(evaluate_one_set("normal", normal_data_dir))
all_rows.extend(evaluate_one_set("abnormal", abnormal_data_dir))

df = pd.DataFrame(all_rows)

csv_path = os.path.join(save_root, "stage10_sliding_candidates.csv")
df.to_csv(csv_path, index=False)

print()
print("CSV saved to:", csv_path)

if len(df) > 0:
    summary = df.groupby(["set_name", "rank"]).agg(
        score_mean=("score", "mean"),
        score_std=("score", "std"),
        region_gain_mean=("region_gain", "mean"),
        region_gain_std=("region_gain", "std"),
        outside_gain_mean=("outside_gain", "mean"),
        outside_gain_std=("outside_gain", "std"),
        ordinary_whole_diff_mean=("ordinary_whole_diff", "mean"),
        perplexity_mean=("perplexity", "mean"),
        n=("case_name", "count")
    ).reset_index()

    summary_csv_path = os.path.join(save_root, "stage10_sliding_candidates_summary.csv")
    summary.to_csv(summary_csv_path, index=False)

    print("Summary CSV saved to:", summary_csv_path)
    print()
    print(summary.to_string(index=False))
else:
    print("No candidate rows were generated.")

print()
print("=" * 80)
print("Saved images under:", save_root)
print("Image columns:")
print("1  input")
print("2  ordinary recon")
print("3  ordinary diff")
print("4  top1 overlay")
print("5  top1 masked input")
print("6  top1 recon")
print("7  top1 diff")
print("8  top2 overlay")
print("9  top2 masked input")
print("10 top2 recon")
print("11 top2 diff")
print("12 top3 overlay")
print("13 top3 masked input")
print("14 top3 recon")
print("15 top3 diff")
print("=" * 80)