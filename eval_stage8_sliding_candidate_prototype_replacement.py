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

abnormal_dir = "/app/data/temp/abnormal"
normal_train_dir = "/app/data/train"

model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage1_full_final.pth"

save_root = "/app/results/eval_stage8_sliding_candidate_prototype_replacement"
os.makedirs(save_root, exist_ok=True)

resize = 256

# ------------------------------------------------------------
# Image-level candidate search
# ------------------------------------------------------------
image_block_size = 40
image_stride = 16
fill_value = 0.08
min_lung_ratio = 0.25

candidate_top_n = 5
candidate_nms_iou = 0.30

score_mode = "topk"
topk_ratio = 0.10

# ------------------------------------------------------------
# Prototype replacement
# ------------------------------------------------------------
max_normal_samples = 500
prototype_top_k = 5
prototype_vis_k = 3

alpha_list = [0.5, 0.7, 1.0]
vis_alpha = 1.0

context_pad = 4

# save images
max_save_images = None


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
# IO utilities
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
# Model helpers
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
# Lung mask and image-level sliding candidate search
# ============================================================

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


def topk_mean(x, ratio=0.10):
    flat = x.reshape(-1)
    k = max(1, int(flat.numel() * ratio))
    vals, _ = torch.topk(flat, k)
    return vals.mean()


def compute_block_score(block_diff):
    if score_mode == "topk":
        return topk_mean(block_diff, ratio=topk_ratio).item()
    elif score_mode == "mean":
        return block_diff.mean().item()
    else:
        raise ValueError(f"Unknown score_mode: {score_mode}")


def box_iou(c1, c2):
    yy1 = max(c1["y1"], c2["y1"])
    yy2 = min(c1["y2"], c2["y2"])
    xx1 = max(c1["x1"], c2["x1"])
    xx2 = min(c1["x2"], c2["x2"])

    inter_h = max(0, yy2 - yy1)
    inter_w = max(0, xx2 - xx1)
    inter = inter_h * inter_w

    area1 = (c1["y2"] - c1["y1"]) * (c1["x2"] - c1["x1"])
    area2 = (c2["y2"] - c2["y1"]) * (c2["x2"] - c2["x1"])

    union = area1 + area2 - inter + 1e-8

    return inter / union


def nms_candidates(candidates, top_n=5, iou_threshold=0.30):
    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)

    selected = []
    for c in candidates:
        keep = True
        for s in selected:
            if box_iou(c, s) > iou_threshold:
                keep = False
                break

        if keep:
            selected.append(c)

        if len(selected) >= top_n:
            break

    return selected


def image_candidate_search(model, x):
    """
    用 image-level sliding mask 找 candidate。
    注意：这里只用来找 candidate，不作为最终 reconstruction。
    """
    lung_mask = make_lung_mask(x)

    response_map = torch.zeros_like(x)
    response_count = torch.zeros_like(x)

    candidates = []

    with torch.no_grad():
        for y1 in range(0, resize - image_block_size + 1, image_stride):
            for x1 in range(0, resize - image_block_size + 1, image_stride):

                y2 = y1 + image_block_size
                x2 = x1 + image_block_size

                block_mask = torch.zeros_like(x)
                block_mask[:, :, y1:y2, x1:x2] = 1.0

                lung_ratio = (
                    lung_mask[:, :, y1:y2, x1:x2].mean().item()
                )

                if lung_ratio < min_lung_ratio:
                    continue

                x_masked = x.clone()
                x_masked[:, :, y1:y2, x1:x2] = fill_value

                recon_masked, _, _, _ = extract_zq_recon_qdist(
                    model,
                    x_masked
                )

                diff = torch.abs(x - recon_masked)

                block_diff = diff[:, :, y1:y2, x1:x2]
                score = compute_block_score(block_diff)

                candidates.append({
                    "score": score,
                    "y1": y1,
                    "y2": y2,
                    "x1": x1,
                    "x2": x2,
                    "lung_ratio": lung_ratio,
                })

                response_map[:, :, y1:y2, x1:x2] += score
                response_count[:, :, y1:y2, x1:x2] += 1.0

    if len(candidates) == 0:
        print("Warning: no lung candidates found. Fallback to all image blocks.")

        with torch.no_grad():
            for y1 in range(0, resize - image_block_size + 1, image_stride):
                for x1 in range(0, resize - image_block_size + 1, image_stride):

                    y2 = y1 + image_block_size
                    x2 = x1 + image_block_size

                    x_masked = x.clone()
                    x_masked[:, :, y1:y2, x1:x2] = fill_value

                    recon_masked, _, _, _ = extract_zq_recon_qdist(
                        model,
                        x_masked
                    )

                    diff = torch.abs(x - recon_masked)
                    block_diff = diff[:, :, y1:y2, x1:x2]
                    score = compute_block_score(block_diff)

                    candidates.append({
                        "score": score,
                        "y1": y1,
                        "y2": y2,
                        "x1": x1,
                        "x2": x2,
                        "lung_ratio": 0.0,
                    })

                    response_map[:, :, y1:y2, x1:x2] += score
                    response_count[:, :, y1:y2, x1:x2] += 1.0

    response_map = response_map / (response_count + 1e-8)

    selected = nms_candidates(
        candidates,
        top_n=candidate_top_n,
        iou_threshold=candidate_nms_iou
    )

    return selected, response_map, lung_mask


# ============================================================
# Coordinate conversion
# ============================================================

def image_box_to_latent_box(y1, y2, x1, x2, pad_latent=1):
    """
    image 256 -> latent 64
    scale = 4
    """
    ly1 = max(0, int(np.floor(y1 / 4)) - pad_latent)
    ly2 = min(64, int(np.ceil(y2 / 4)) + pad_latent)

    lx1 = max(0, int(np.floor(x1 / 4)) - pad_latent)
    lx2 = min(64, int(np.ceil(x2 / 4)) + pad_latent)

    return ly1, ly2, lx1, lx2


def latent_to_image_box(ly1, ly2, lx1, lx2):
    y1 = ly1 * 4
    y2 = ly2 * 4
    x1 = lx1 * 4
    x2 = lx2 * 4

    return y1, y2, x1, x2


def make_image_mask_from_candidate(x, candidate):
    mask = torch.zeros_like(x)
    mask[:, :, candidate["y1"]:candidate["y2"], candidate["x1"]:candidate["x2"]] = 1.0
    return mask


def make_union_mask(x, candidates):
    mask = torch.zeros_like(x)
    for c in candidates:
        mask[:, :, c["y1"]:c["y2"], c["x1"]:c["x2"]] = 1.0
    return mask


def make_overlay(x, mask):
    overlay = x.clone() * 0.75 + mask * 0.25
    overlay = torch.clamp(overlay, 0.0, 1.0)
    return overlay


# ============================================================
# Prototype bank
# ============================================================

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


def build_normal_zq_bank(model, normal_paths):
    bank = []

    print()
    print("Building normal z_q bank...")
    print("Normal samples:", len(normal_paths))

    with torch.no_grad():
        for i, path in enumerate(normal_paths):
            try:
                x_n = load_dicom_as_tensor(path).to(device)
                _, zq_n, _, _ = extract_zq_recon_qdist(model, x_n)

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


def search_prototypes(target_zq, normal_bank, ly1, ly2, lx1, lx2):
    candidates = []

    for item in normal_bank:
        normal_zq = item["zq"].to(device)

        mse = compute_context_mse(
            target_zq,
            normal_zq,
            ly1,
            ly2,
            lx1,
            lx2,
            context_pad
        )

        candidates.append({
            "path": item["path"],
            "case_name": item["case_name"],
            "zq": item["zq"],
            "context_mse": mse,
        })

    candidates = sorted(candidates, key=lambda c: c["context_mse"])
    return candidates[:prototype_top_k]


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


def apply_candidate_prototype_replacement(
    z_q,
    candidate_infos,
    alpha,
    prototype_rank=0
):
    """
    对 top1/top3/top5 candidates 做 prototype replacement。
    每个 candidate 使用自己 context-matched 的 prototype。
    """
    z_corr = z_q.clone()

    for info in candidate_infos:
        proto = info["prototypes"][prototype_rank]
        proto_zq = proto["zq"].to(device)

        ly1 = info["ly1"]
        ly2 = info["ly2"]
        lx1 = info["lx1"]
        lx2 = info["lx2"]

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
# Build normal bank
# ============================================================

normal_paths = collect_dicom_paths(
    normal_train_dir,
    max_samples=max_normal_samples
)

normal_bank = build_normal_zq_bank(model, normal_paths)


# ============================================================
# Evaluate one case
# ============================================================

def evaluate_case(case_idx, dicom_path):
    case_name = os.path.basename(dicom_path).replace(".dcm", "")

    case_save_dir = os.path.join(save_root, case_name)
    os.makedirs(case_save_dir, exist_ok=True)

    x = load_dicom_as_tensor(dicom_path).to(device)

    with torch.no_grad():
        recon_original, z_q, q_distance_map, perplexity = extract_zq_recon_qdist(
            model,
            x
        )

        diff_original = torch.abs(x - recon_original)

        q_map = q_distance_map.unsqueeze(1)
        q_map = F.interpolate(
            q_map,
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

    print()
    print("=" * 80)
    print(f"Case {case_idx}: {case_name}")
    print("ordinary_diff:", diff_original.mean().item())

    # --------------------------------------------------------
    # Step 1: image-level candidate search
    # --------------------------------------------------------
    candidates, response_map, lung_mask = image_candidate_search(model, x)

    print("Top image candidates:")
    for i, c in enumerate(candidates):
        print(
            f"top{i+1}: score={c['score']:.6f}, "
            f"y={c['y1']}:{c['y2']}, x={c['x1']}:{c['x2']}, "
            f"lung={c['lung_ratio']:.3f}"
        )

    # --------------------------------------------------------
    # Step 2: convert candidates to latent boxes and search prototypes
    # --------------------------------------------------------
    candidate_infos = []

    for i, c in enumerate(candidates):
        ly1, ly2, lx1, lx2 = image_box_to_latent_box(
            c["y1"],
            c["y2"],
            c["x1"],
            c["x2"],
            pad_latent=1
        )

        prototypes = search_prototypes(
            z_q.detach(),
            normal_bank,
            ly1,
            ly2,
            lx1,
            lx2
        )

        info = dict(c)
        info["candidate_rank"] = i + 1
        info["ly1"] = ly1
        info["ly2"] = ly2
        info["lx1"] = lx1
        info["lx2"] = lx2
        info["prototypes"] = prototypes

        candidate_infos.append(info)

        print(
            f"candidate {i+1}: latent y={ly1}:{ly2}, x={lx1}:{lx2}, "
            f"proto1={prototypes[0]['case_name']}, mse={prototypes[0]['context_mse']:.6f}"
        )

    records = []

    # --------------------------------------------------------
    # Step 3: reconstruct top1/top3/top5 union candidate replacement
    # --------------------------------------------------------
    for n_candidate in [1, 3, 5]:
        n_candidate = min(n_candidate, len(candidate_infos))
        selected_infos = candidate_infos[:n_candidate]
        selected_candidates = candidates[:n_candidate]

        union_mask = make_union_mask(x, selected_candidates)

        ordinary_region_diff = (
            diff_original * union_mask
        ).sum().item() / (union_mask.sum().item() + 1e-8)

        ordinary_outside_diff = (
            diff_original * (1.0 - union_mask)
        ).sum().item() / ((1.0 - union_mask).sum().item() + 1e-8)

        for alpha in alpha_list:
            with torch.no_grad():
                z_corr = apply_candidate_prototype_replacement(
                    z_q,
                    selected_infos,
                    alpha=alpha,
                    prototype_rank=0
                )

                recon_corr = model.decoder(z_corr)

                diff_corr, whole_diff, region_diff, outside_diff = compute_scores(
                    x,
                    recon_corr,
                    union_mask
                )

            records.append({
                "case_name": case_name,
                "path": dicom_path,

                "n_candidate": n_candidate,
                "alpha": alpha,

                "ordinary_whole_diff": diff_original.mean().item(),
                "ordinary_region_diff": ordinary_region_diff,
                "ordinary_outside_diff": ordinary_outside_diff,

                "whole_diff": whole_diff,
                "region_diff": region_diff,
                "outside_diff": outside_diff,

                "region_gain_vs_ordinary": region_diff - ordinary_region_diff,
                "outside_gain_vs_ordinary": outside_diff - ordinary_outside_diff,
                "gain_ratio": (
                    (region_diff - ordinary_region_diff)
                    / (outside_diff - ordinary_outside_diff + 1e-8)
                ),

                "perplexity": float(perplexity.item()),
            })

    # --------------------------------------------------------
    # Save visualization image
    # --------------------------------------------------------
    should_save = True
    if max_save_images is not None and case_idx >= max_save_images:
        should_save = False

    if should_save:
        masks = []
        recons = []
        diffs = []

        for n_candidate in [1, 3, 5]:
            n_candidate = min(n_candidate, len(candidate_infos))
            selected_infos = candidate_infos[:n_candidate]
            selected_candidates = candidates[:n_candidate]

            union_mask = make_union_mask(x, selected_candidates)

            with torch.no_grad():
                z_corr = apply_candidate_prototype_replacement(
                    z_q,
                    selected_infos,
                    alpha=vis_alpha,
                    prototype_rank=0
                )

                recon_corr = model.decoder(z_corr)
                diff_corr = torch.abs(x - recon_corr)

            masks.append(union_mask.cpu())
            recons.append(recon_corr.cpu())
            diffs.append(normalize_map(diff_corr).cpu())

        top1_overlay = make_overlay(x, masks[0].to(device)).cpu()

        grid_items = [
            x.cpu(),                                  # 1 input
            recon_original.cpu(),                     # 2 ordinary recon
            normalize_map(diff_original).cpu(),       # 3 ordinary diff
            lung_mask.cpu(),                          # 4 lung mask
            normalize_map(response_map).cpu(),        # 5 image response map
            top1_overlay,                             # 6 top1 overlay
            masks[0],                                 # 7 top1 mask
            recons[0],                                # 8 top1 proto recon
            diffs[0],                                 # 9 top1 proto diff
            masks[1],                                 # 10 top3 mask
            recons[1],                                # 11 top3 proto recon
            diffs[1],                                 # 12 top3 proto diff
            masks[2],                                 # 13 top5 mask
            recons[2],                                # 14 top5 proto recon
            diffs[2],                                 # 15 top5 proto diff
            normalize_map(q_map).cpu(),               # 16 qdistance
        ]

        grid = vutils.make_grid(
            torch.cat(grid_items, dim=0),
            nrow=16,
            normalize=True
        )

        save_img_path = os.path.join(
            case_save_dir,
            f"{case_name}_stage8_candidate_prototype_replacement.png"
        )

        vutils.save_image(grid, save_img_path)
        print("Saved image:", save_img_path)

    # --------------------------------------------------------
    # Candidate details
    # --------------------------------------------------------
    for info in candidate_infos:
        c_rank = info["candidate_rank"]

        for p_rank, proto in enumerate(info["prototypes"]):
            records.append({
                "case_name": case_name,
                "path": dicom_path,

                "record_type": "candidate_prototype",
                "candidate_rank": c_rank,

                "candidate_score": info["score"],
                "candidate_y1": info["y1"],
                "candidate_y2": info["y2"],
                "candidate_x1": info["x1"],
                "candidate_x2": info["x2"],

                "latent_y1": info["ly1"],
                "latent_y2": info["ly2"],
                "latent_x1": info["lx1"],
                "latent_x2": info["lx2"],

                "prototype_rank": p_rank + 1,
                "prototype_case": proto["case_name"],
                "prototype_path": proto["path"],
                "context_mse": proto["context_mse"],
            })

    return records


# ============================================================
# Run
# ============================================================

abnormal_paths = collect_dicom_paths(abnormal_dir)

print()
print("Abnormal cases:", len(abnormal_paths))

all_records = []

for idx, path in enumerate(abnormal_paths):
    case_records = evaluate_case(idx, path)
    all_records.extend(case_records)

df = pd.DataFrame(all_records)

csv_path = os.path.join(
    save_root,
    "stage8_sliding_candidate_prototype_replacement_scores.csv"
)

df.to_csv(csv_path, index=False)

print()
print("Saved CSV:", csv_path)


# ============================================================
# Summary
# ============================================================

score_df = df[df.get("record_type").isna()].copy()

summary = (
    score_df
    .groupby(["n_candidate", "alpha"])
    .agg(
        region_gain_mean=("region_gain_vs_ordinary", "mean"),
        region_gain_std=("region_gain_vs_ordinary", "std"),
        outside_gain_mean=("outside_gain_vs_ordinary", "mean"),
        outside_gain_std=("outside_gain_vs_ordinary", "std"),
        gain_ratio_mean=("gain_ratio", "mean"),
        region_diff_mean=("region_diff", "mean"),
        outside_diff_mean=("outside_diff", "mean"),
        n=("case_name", "count"),
    )
    .reset_index()
)

summary_path = os.path.join(
    save_root,
    "stage8_summary_by_topn_alpha.csv"
)

summary.to_csv(summary_path, index=False)

print()
print("Summary:")
print(summary.to_string(index=False))

print()
print("Saved summary CSV:", summary_path)

print()
print("Image columns:")
print("1 input")
print("2 ordinary recon")
print("3 ordinary diff")
print("4 lung mask")
print("5 image response map")
print("6 top1 overlay")
print("7 top1 mask")
print("8 top1 prototype recon")
print("9 top1 prototype diff")
print("10 top3 mask")
print("11 top3 prototype recon")
print("12 top3 prototype diff")
print("13 top5 mask")
print("14 top5 prototype recon")
print("15 top5 prototype diff")
print("16 qdistance")