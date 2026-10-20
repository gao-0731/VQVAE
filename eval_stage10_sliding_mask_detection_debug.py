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

save_root = "/app/results/eval_stage10_sliding_mask_detection_debug"
os.makedirs(save_root, exist_ok=True)

resize = 256

# ------------------------------------------------------------
# Sliding mask settings
# ------------------------------------------------------------
mask_size_list = [32, 40, 48]
stride = 8

# 保存/显示多少
top_n_for_image = 10
top_n_for_csv = 50

nms_iou_thr = 0.30

fill_mode = "local_mean"   # "local_mean" or "constant"
fill_value = 0.08
local_mean_pad = 8

min_lung_ratio = 0.20

score_mode = "topk"        # "topk" or "mean"
topk_ratio = 0.10

# 用哪个 score 做排序
# 可选: "score_penalty", "score_raw", "score_ratio", "score_gain"
ranking_score_name = "score_penalty"

outside_penalty_weight = 0.30

# limit optional
max_cases_per_set = None   # None means all

# ------------------------------------------------------------
# Manual lesion debug
# ------------------------------------------------------------
manual_debug = True

# 你之前 512 坐标:
# left top     = (130, 340)
# right bottom = (200, 380)
# resize 到 256 后大概:
# x = 65~100
# y = 170~190
manual_center_y = 180
manual_center_x = 82

manual_debug_sizes = [24, 32, 40, 48]


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
    x: [1,1,256,256]
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
    vals = diff[mask > 0.5]

    if vals.numel() == 0:
        return 0.0

    if mode == "mean":
        return vals.mean().item()

    if mode == "topk":
        k = max(1, int(math.ceil(vals.numel() * topk_ratio)))
        topv, _ = torch.topk(vals, k)
        return topv.mean().item()

    raise ValueError(f"Unknown score mode: {mode}")


def compute_mean_outside_score(diff, mask):
    vals = diff[mask < 0.5]

    if vals.numel() == 0:
        return 0.0

    return vals.mean().item()


def box_iou(box_a, box_b):
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


def nms_candidates(candidates, score_name, iou_thr=0.30, top_n=10):
    candidates = sorted(
        candidates,
        key=lambda x: x[score_name],
        reverse=True
    )

    selected = []

    for cand in candidates:
        keep = True

        for sel in selected:
            if box_iou(cand["box"], sel["box"]) > iou_thr:
                keep = False
                break

        if keep:
            selected.append(cand)

        if len(selected) >= top_n:
            break

    return selected


def get_box_from_center(cy, cx, size):
    half = size // 2

    y1 = max(0, cy - half)
    y2 = min(resize, cy + half)

    x1 = max(0, cx - half)
    x2 = min(resize, cx + half)

    if y2 - y1 < size:
        if y1 == 0:
            y2 = min(resize, size)
        elif y2 == resize:
            y1 = max(0, resize - size)

    if x2 - x1 < size:
        if x1 == 0:
            x2 = min(resize, size)
        elif x2 == resize:
            x1 = max(0, resize - size)

    return y1, y2, x1, x2


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
# Candidate scoring
# ============================================================

def score_one_box(model, x, ordinary_diff, y1, y2, x1, x2, mask_size):
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

    score_raw = masked_region_score
    score_penalty = masked_region_score - outside_penalty_weight * masked_outside_score
    score_ratio = masked_region_score / (masked_outside_score + 1e-8)
    score_gain = region_gain - outside_penalty_weight * max(0.0, outside_gain)

    row = {
        "box": (y1, y2, x1, x2),
        "mask_size": mask_size,
        "y1": y1,
        "y2": y2,
        "x1": x1,
        "x2": x2,

        "ordinary_region_score": ordinary_region_score,
        "masked_region_score": masked_region_score,
        "ordinary_outside_score": ordinary_outside_score,
        "masked_outside_score": masked_outside_score,

        "region_gain": region_gain,
        "outside_gain": outside_gain,

        "score_raw": score_raw,
        "score_penalty": score_penalty,
        "score_ratio": score_ratio,
        "score_gain": score_gain,

        "perplexity": float(perplexity.item()),
    }

    return row, x_masked, recon, diff, mask


def evaluate_manual_debug_boxes(model, x, ordinary_diff, all_candidates_sorted):
    rows = []

    print()
    print("Manual lesion debug scores:")

    for mask_size in manual_debug_sizes:
        y1, y2, x1, x2 = get_box_from_center(
            manual_center_y,
            manual_center_x,
            mask_size
        )

        row, _, _, _, _ = score_one_box(
            model,
            x,
            ordinary_diff,
            y1,
            y2,
            x1,
            x2,
            mask_size
        )

        manual_box = row["box"]

        # rank by max IoU with sliding candidates
        best_iou = -1.0
        best_rank = None
        best_candidate = None

        for rank, cand in enumerate(all_candidates_sorted, start=1):
            cur_iou = box_iou(manual_box, cand["box"])
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_rank = rank
                best_candidate = cand

        row["manual_debug"] = True
        row["manual_best_matching_rank"] = best_rank
        row["manual_best_matching_iou"] = best_iou

        if best_candidate is not None:
            row["manual_best_matching_score"] = best_candidate[ranking_score_name]
            row["manual_best_matching_y1"] = best_candidate["y1"]
            row["manual_best_matching_y2"] = best_candidate["y2"]
            row["manual_best_matching_x1"] = best_candidate["x1"]
            row["manual_best_matching_x2"] = best_candidate["x2"]
            row["manual_best_matching_mask_size"] = best_candidate["mask_size"]

        print(
            f"manual mask={mask_size}, "
            f"box y={y1}:{y2}, x={x1}:{x2}, "
            f"raw={row['score_raw']:.6f}, "
            f"penalty={row['score_penalty']:.6f}, "
            f"ratio={row['score_ratio']:.3f}, "
            f"gain={row['score_gain']:.6f}, "
            f"best_match_rank={best_rank}, "
            f"best_iou={best_iou:.3f}"
        )

        rows.append(row)

    return rows


# ============================================================
# Generate candidates
# ============================================================

def generate_candidates_for_one_image(model, x):
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

                row, _, _, _, _ = score_one_box(
                    model,
                    x,
                    ordinary_diff,
                    y1,
                    y2,
                    x1,
                    x2,
                    block_size
                )

                row["lung_ratio"] = lung_ratio
                all_candidates.append(row)

    all_candidates_sorted = sorted(
        all_candidates,
        key=lambda x: x[ranking_score_name],
        reverse=True
    )

    for rank, cand in enumerate(all_candidates_sorted, start=1):
        cand["rank_all"] = rank

    selected_candidates = nms_candidates(
        all_candidates,
        score_name=ranking_score_name,
        iou_thr=nms_iou_thr,
        top_n=top_n_for_image
    )

    for rank, cand in enumerate(selected_candidates, start=1):
        cand["rank_nms"] = rank

    manual_rows = []
    if manual_debug:
        manual_rows = evaluate_manual_debug_boxes(
            model,
            x,
            ordinary_diff,
            all_candidates_sorted
        )

    return (
        ordinary_recon,
        ordinary_diff,
        ordinary_perplexity,
        selected_candidates,
        all_candidates_sorted,
        manual_rows
    )


# ============================================================
# Visualization
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

        row, x_masked, recon, diff, mask = score_one_box(
            model,
            x,
            ordinary_diff,
            y1,
            y2,
            x1,
            x2,
            cand["mask_size"]
        )

        overlay = make_overlay(x, mask)

        grid_items.extend([
            overlay.cpu(),
            x_masked.cpu(),
            recon.cpu(),
            normalize_map(diff).cpu(),
        ])

        rec = dict(cand)
        rec.pop("box", None)
        rec["rank_image"] = rank
        candidate_records.append(rec)

    ncol = 3 + 4 * len(selected_candidates)

    grid = vutils.make_grid(
        torch.cat(grid_items, dim=0),
        nrow=ncol,
        normalize=True
    )

    save_img_path = os.path.join(
        case_save_dir,
        f"{case_name}_top{len(selected_candidates)}_{ranking_score_name}.png"
    )

    vutils.save_image(grid, save_img_path)

    return save_img_path, candidate_records


# ============================================================
# Evaluate set
# ============================================================

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

    top_rows = []
    all_candidate_rows = []
    manual_debug_rows = []

    for idx, dicom_path in enumerate(paths):
        case_name = os.path.basename(dicom_path).replace(".dcm", "")

        case_save_dir = os.path.join(set_save_dir, case_name)
        os.makedirs(case_save_dir, exist_ok=True)

        print()
        print(f"[{set_name}] {idx+1}/{len(paths)}  {case_name}")

        x = load_dicom_lung_window(dicom_path).to(device)

        (
            ordinary_recon,
            ordinary_diff,
            ordinary_perplexity,
            selected_candidates,
            all_candidates_sorted,
            manual_rows
        ) = generate_candidates_for_one_image(model, x)

        save_img_path, candidate_records = save_case_visualization(
            x,
            ordinary_recon,
            ordinary_diff,
            selected_candidates,
            case_save_dir,
            case_name
        )

        # save top rows
        for row in candidate_records:
            row["set_name"] = set_name
            row["case_name"] = case_name
            row["path"] = dicom_path
            row["ordinary_whole_diff"] = ordinary_diff.mean().item()
            row["ordinary_perplexity"] = float(ordinary_perplexity.item())
            row["image_path"] = save_img_path
            top_rows.append(row)

        # save top50 all candidates, before NMS
        for cand in all_candidates_sorted[:top_n_for_csv]:
            row = dict(cand)
            row.pop("box", None)
            row["set_name"] = set_name
            row["case_name"] = case_name
            row["path"] = dicom_path
            row["ordinary_whole_diff"] = ordinary_diff.mean().item()
            row["ordinary_perplexity"] = float(ordinary_perplexity.item())
            all_candidate_rows.append(row)

        # save manual debug rows
        for mrow in manual_rows:
            row = dict(mrow)
            row.pop("box", None)
            row["set_name"] = set_name
            row["case_name"] = case_name
            row["path"] = dicom_path
            row["ordinary_whole_diff"] = ordinary_diff.mean().item()
            row["ordinary_perplexity"] = float(ordinary_perplexity.item())
            manual_debug_rows.append(row)

        print("  saved:", save_img_path)

        if len(candidate_records) > 0:
            best = candidate_records[0]
            print(
                f"  top1 {ranking_score_name}={best[ranking_score_name]:.6f}, "
                f"raw={best['score_raw']:.6f}, "
                f"penalty={best['score_penalty']:.6f}, "
                f"ratio={best['score_ratio']:.3f}, "
                f"gain={best['score_gain']:.6f}, "
                f"mask={best['mask_size']}, "
                f"box=({best['y1']}:{best['y2']}, {best['x1']}:{best['x2']})"
            )

    return top_rows, all_candidate_rows, manual_debug_rows


# ============================================================
# Run
# ============================================================

all_top_rows = []
all_candidate_rows = []
all_manual_rows = []

rows_top, rows_all, rows_manual = evaluate_one_set("normal", normal_data_dir)
all_top_rows.extend(rows_top)
all_candidate_rows.extend(rows_all)
all_manual_rows.extend(rows_manual)

rows_top, rows_all, rows_manual = evaluate_one_set("abnormal", abnormal_data_dir)
all_top_rows.extend(rows_top)
all_candidate_rows.extend(rows_all)
all_manual_rows.extend(rows_manual)


# ============================================================
# Save CSVs
# ============================================================

top_df = pd.DataFrame(all_top_rows)
top_csv_path = os.path.join(
    save_root,
    f"stage10_top{top_n_for_image}_nms_candidates_{ranking_score_name}.csv"
)
top_df.to_csv(top_csv_path, index=False)

all_df = pd.DataFrame(all_candidate_rows)
all_csv_path = os.path.join(
    save_root,
    f"stage10_top{top_n_for_csv}_all_candidates_before_nms_{ranking_score_name}.csv"
)
all_df.to_csv(all_csv_path, index=False)

manual_df = pd.DataFrame(all_manual_rows)
manual_csv_path = os.path.join(
    save_root,
    f"stage10_manual_debug_scores_{ranking_score_name}.csv"
)
manual_df.to_csv(manual_csv_path, index=False)

print()
print("=" * 80)
print("Saved CSV files:")
print("Top NMS candidates:", top_csv_path)
print("Top all candidates before NMS:", all_csv_path)
print("Manual debug:", manual_csv_path)


# ============================================================
# Summary
# ============================================================

if len(top_df) > 0:
    summary = top_df.groupby(["set_name", "rank_image"]).agg(
        score_raw_mean=("score_raw", "mean"),
        score_raw_std=("score_raw", "std"),
        score_penalty_mean=("score_penalty", "mean"),
        score_penalty_std=("score_penalty", "std"),
        score_ratio_mean=("score_ratio", "mean"),
        score_ratio_std=("score_ratio", "std"),
        score_gain_mean=("score_gain", "mean"),
        score_gain_std=("score_gain", "std"),
        masked_region_score_mean=("masked_region_score", "mean"),
        masked_region_score_std=("masked_region_score", "std"),
        region_gain_mean=("region_gain", "mean"),
        region_gain_std=("region_gain", "std"),
        masked_outside_score_mean=("masked_outside_score", "mean"),
        outside_gain_mean=("outside_gain", "mean"),
        ordinary_whole_diff_mean=("ordinary_whole_diff", "mean"),
        n=("case_name", "count"),
    ).reset_index()

    summary_csv_path = os.path.join(
        save_root,
        f"stage10_top{top_n_for_image}_summary_{ranking_score_name}.csv"
    )
    summary.to_csv(summary_csv_path, index=False)

    print("Summary:", summary_csv_path)
    print()
    print(summary.to_string(index=False))

if len(manual_df) > 0:
    manual_summary = manual_df.groupby(["set_name", "mask_size"]).agg(
        score_raw_mean=("score_raw", "mean"),
        score_penalty_mean=("score_penalty", "mean"),
        score_ratio_mean=("score_ratio", "mean"),
        score_gain_mean=("score_gain", "mean"),
        masked_region_score_mean=("masked_region_score", "mean"),
        region_gain_mean=("region_gain", "mean"),
        outside_gain_mean=("outside_gain", "mean"),
        best_matching_rank_mean=("manual_best_matching_rank", "mean"),
        best_matching_iou_mean=("manual_best_matching_iou", "mean"),
        n=("case_name", "count"),
    ).reset_index()

    manual_summary_csv_path = os.path.join(
        save_root,
        f"stage10_manual_debug_summary_{ranking_score_name}.csv"
    )
    manual_summary.to_csv(manual_summary_csv_path, index=False)

    print()
    print("Manual debug summary:", manual_summary_csv_path)
    print(manual_summary.to_string(index=False))


print()
print("=" * 80)
print("Finished.")
print("Saved images under:", save_root)
print()
print("Image columns:")
print("1 input")
print("2 ordinary recon")
print("3 ordinary diff")
for i in range(1, top_n_for_image + 1):
    base = 4 + (i - 1) * 4
    print(f"{base} top{i} overlay")
    print(f"{base+1} top{i} masked input")
    print(f"{base+2} top{i} recon")
    print(f"{base+3} top{i} diff")
print("=" * 80)