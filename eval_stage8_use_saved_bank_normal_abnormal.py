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

# test data
normal_test_dir = "/app/data/temp/normal"
abnormal_dir = "/app/data/temp/abnormal"

# Stage1 full VQ-VAE
model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage1_full_final.pth"

# saved normal prototype bank
normal_bank_path = "/app/results/normal_zq_bank_stage1_random5000.pt"

save_root = "/app/results/eval_stage8_savedbank_random5000_normal_abnormal24"
os.makedirs(save_root, exist_ok=True)

resize = 256

# ------------------------------------------------------------
# Image-level candidate search
# ------------------------------------------------------------
image_block_size = 24
image_stride = 16
fill_value = 0.08
min_lung_ratio = 0

candidate_top_n = 5
candidate_nms_iou = 0.30

score_mode = "topk"
topk_ratio = 0.10

# ------------------------------------------------------------
# Prototype replacement
# ------------------------------------------------------------
prototype_top_k = 5
prototype_vis_k = 3

alpha_list = [0.5, 0.7, 1.0]
vis_alpha = 0.7

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

def collect_dicom_paths(root_dir):
    paths = []

    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".dcm"):
                paths.append(os.path.join(root, f))

    paths = sorted(paths)

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
# Lung mask and image-level candidate search
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
    Image-level sliding mask candidate search.
    This is only used to generate top1/top3/top5 candidates.
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

                lung_ratio = lung_mask[:, :, y1:y2, x1:x2].mean().item()

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
# Prototype bank helpers
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


def apply_candidate_prototype_replacement(
    z_q,
    candidate_infos,
    alpha,
    prototype_rank=0
):
    """
    For top1/top3/top5 candidates.
    Each candidate uses its own context-matched normal prototype.
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
# Load saved normal z_q bank
# ============================================================

print()
print("Loading saved normal z_q bank:")
print(normal_bank_path)

bank_data = torch.load(normal_bank_path, map_location="cpu")

normal_bank = bank_data["bank"]

print("Loaded normal bank size:", len(normal_bank))
print("Bank model path:", bank_data.get("model_path", "unknown"))
print("Bank random seed:", bank_data.get("random_seed", "unknown"))
print("Bank samples saved:", bank_data.get("n_samples_saved", "unknown"))


# ============================================================
# Evaluate one case
# ============================================================

def evaluate_case(case_idx, dicom_path, label):
    case_name = os.path.basename(dicom_path).replace(".dcm", "")

    case_save_dir = os.path.join(save_root, label, case_name)
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
    print(f"[{label}] Case {case_idx}: {case_name}")
    print("ordinary_diff:", diff_original.mean().item())

    # --------------------------------------------------------
    # Step 1: image-level candidate search
    # --------------------------------------------------------
    candidates, response_map, lung_mask = image_candidate_search(
        model,
        x
    )

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
                "label": label,
                "case_name": case_name,
                "path": dicom_path,

                "record_type": "summary_score",

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
    # Candidate details
    # --------------------------------------------------------
    for info in candidate_infos:
        c_rank = info["candidate_rank"]

        for p_rank, proto in enumerate(info["prototypes"]):
            records.append({
                "label": label,
                "case_name": case_name,
                "path": dicom_path,

                "record_type": "candidate_prototype",

                "candidate_rank": c_rank,
                "candidate_score": info["score"],

                "candidate_y1": info["y1"],
                "candidate_y2": info["y2"],
                "candidate_x1": info["x1"],
                "candidate_x2": info["x2"],
                "candidate_lung_ratio": info["lung_ratio"],

                "latent_y1": info["ly1"],
                "latent_y2": info["ly2"],
                "latent_x1": info["lx1"],
                "latent_x2": info["lx2"],

                "prototype_rank": p_rank + 1,
                "prototype_case": proto["case_name"],
                "prototype_path": proto["path"],
                "context_mse": proto["context_mse"],
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
        top3_overlay = make_overlay(x, masks[1].to(device)).cpu()

        grid_items = [
            x.cpu(),                                  # 1 input
            recon_original.cpu(),                     # 2 ordinary recon
            normalize_map(diff_original).cpu(),       # 3 ordinary diff
            lung_mask.cpu(),                          # 4 lung mask
            normalize_map(response_map).cpu(),        # 5 image response map

            top1_overlay,                             # 6 top1 overlay
            masks[0],                                 # 7 top1 mask
            recons[0],                                # 8 top1 prototype recon
            diffs[0],                                 # 9 top1 prototype diff

            top3_overlay,                             # 10 top3 overlay
            masks[1],                                 # 11 top3 mask
            recons[1],                                # 12 top3 prototype recon
            diffs[1],                                 # 13 top3 prototype diff

            masks[2],                                 # 14 top5 mask
            recons[2],                                # 15 top5 prototype recon
            diffs[2],                                 # 16 top5 prototype diff

            normalize_map(q_map).cpu(),               # 17 qdistance
        ]

        grid = vutils.make_grid(
            torch.cat(grid_items, dim=0),
            nrow=17,
            normalize=True
        )

        save_img_path = os.path.join(
            case_save_dir,
            f"{case_name}_stage8_savedbank_random5000.png"
        )

        vutils.save_image(grid, save_img_path)

        print("Saved image:", save_img_path)

    return records


# ============================================================
# Run normal + abnormal
# ============================================================

normal_paths = collect_dicom_paths(normal_test_dir)
abnormal_paths = collect_dicom_paths(abnormal_dir)

print()
print("Normal test cases:", len(normal_paths))
print("Abnormal cases:", len(abnormal_paths))

all_records = []

for idx, path in enumerate(normal_paths):
    records = evaluate_case(idx, path, label="normal")
    all_records.extend(records)

for idx, path in enumerate(abnormal_paths):
    records = evaluate_case(idx, path, label="abnormal")
    all_records.extend(records)


# ============================================================
# Save CSV
# ============================================================

df = pd.DataFrame(all_records)

csv_path = os.path.join(
    save_root,
    "stage8_savedbank_random5000_normal_abnormal_scores.csv"
)

df.to_csv(csv_path, index=False)

print()
print("Saved CSV:", csv_path)


# ============================================================
# Summary by label
# ============================================================

score_df = df[df["record_type"] == "summary_score"].copy()

summary = (
    score_df
    .groupby(["label", "n_candidate", "alpha"])
    .agg(
        region_gain_mean=("region_gain_vs_ordinary", "mean"),
        region_gain_std=("region_gain_vs_ordinary", "std"),
        outside_gain_mean=("outside_gain_vs_ordinary", "mean"),
        outside_gain_std=("outside_gain_vs_ordinary", "std"),
        gain_ratio_mean=("gain_ratio", "mean"),
        region_diff_mean=("region_diff", "mean"),
        region_diff_std=("region_diff", "std"),
        outside_diff_mean=("outside_diff", "mean"),
        whole_diff_mean=("whole_diff", "mean"),
        n=("case_name", "count"),
    )
    .reset_index()
)

summary_path = os.path.join(
    save_root,
    "stage8_summary_by_label_topn_alpha.csv"
)

summary.to_csv(summary_path, index=False)

print()
print("=" * 80)
print("Summary by label / topN / alpha:")
print(summary.to_string(index=False))

print()
print("Saved summary CSV:", summary_path)


# ============================================================
# Detection-style summary using normal threshold
# ============================================================

print()
print("=" * 80)
print("Detection-style summary using normal threshold")
print("Threshold = normal mean + 3 * normal std")

detection_records = []

score_names = [
    "region_gain_vs_ordinary",
    "region_diff",
    "whole_diff",
    "gain_ratio",
]

for score_name in score_names:
    print()
    print(f"[{score_name}]")

    for n_candidate in [1, 3, 5]:
        for alpha in alpha_list:

            sub = score_df[
                (score_df["n_candidate"] == n_candidate)
                & (score_df["alpha"] == alpha)
            ]

            normal_scores = sub[sub["label"] == "normal"][score_name].dropna().values
            abnormal_scores = sub[sub["label"] == "abnormal"][score_name].dropna().values

            if len(normal_scores) == 0 or len(abnormal_scores) == 0:
                continue

            threshold = normal_scores.mean() + 3.0 * normal_scores.std()

            normal_fp = int((normal_scores > threshold).sum())
            abnormal_detected = int((abnormal_scores > threshold).sum())

            fp_rate = normal_fp / len(normal_scores)
            detection_rate = abnormal_detected / len(abnormal_scores)

            print(
                f"top{n_candidate}, alpha={alpha}: "
                f"normal mean={normal_scores.mean():.6f}, "
                f"std={normal_scores.std():.6f}, "
                f"abnormal mean={abnormal_scores.mean():.6f}, "
                f"std={abnormal_scores.std():.6f}, "
                f"threshold={threshold:.6f}, "
                f"FP={normal_fp}/{len(normal_scores)}, "
                f"Detected={abnormal_detected}/{len(abnormal_scores)}"
            )

            detection_records.append({
                "score_name": score_name,
                "n_candidate": n_candidate,
                "alpha": alpha,

                "normal_mean": normal_scores.mean(),
                "normal_std": normal_scores.std(),
                "abnormal_mean": abnormal_scores.mean(),
                "abnormal_std": abnormal_scores.std(),

                "threshold_normal_mean_plus_3std": threshold,

                "normal_false_positive": normal_fp,
                "normal_total": len(normal_scores),
                "normal_fp_rate": fp_rate,

                "abnormal_detected": abnormal_detected,
                "abnormal_total": len(abnormal_scores),
                "abnormal_detection_rate": detection_rate,
            })

detection_df = pd.DataFrame(detection_records)

detection_csv_path = os.path.join(
    save_root,
    "stage8_detection_summary_normal_threshold.csv"
)

detection_df.to_csv(detection_csv_path, index=False)

print()
print("Saved detection summary CSV:", detection_csv_path)


# ============================================================
# Image column explanation
# ============================================================

print()
print("=" * 80)
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
print("10 top3 overlay")
print("11 top3 mask")
print("12 top3 prototype recon")
print("13 top3 prototype diff")
print("14 top5 mask")
print("15 top5 prototype recon")
print("16 top5 prototype diff")
print("17 qdistance")