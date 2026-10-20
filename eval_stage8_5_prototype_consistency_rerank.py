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

save_root = "/app/results/eval_stage8_5_consistency_rerank_random5000"
os.makedirs(save_root, exist_ok=True)

resize = 256

# ------------------------------------------------------------
# Image-level coarse candidate generation
# ------------------------------------------------------------
image_block_size = 40
image_stride = 16
fill_value = 0.08
min_lung_ratio = 0.25

coarse_top_n = 30
candidate_nms_iou = 0.30

score_mode = "topk"
topk_ratio = 0.10

# ------------------------------------------------------------
# Prototype replacement / reranking
# ------------------------------------------------------------
prototype_top_k = 5
alpha_for_rerank = 0.7
alpha_list = [0.5, 0.7, 1.0]
vis_alpha = 0.7

context_pad = 4

# reranking weights
outside_weight = 20.0
prototype_std_weight = 0.5
context_mse_weight = 0.2
monotonic_weight = 0.3

# speed
prototype_search_chunk_size = 256

# image saving
max_save_images = None   # None: save all, 0: save none


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
# Lung mask and image-level coarse candidate search
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


def nms_candidates(candidates, top_n=30, iou_threshold=0.30):
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
    Generate coarse top-N candidates using image-level masked reconstruction response.
    This is only candidate generation, not final decision.
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
        top_n=coarse_top_n,
        iou_threshold=candidate_nms_iou
    )

    return selected, response_map, lung_mask


# ============================================================
# Coordinate conversion and masks
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


def search_prototypes_fast(
    target_zq,
    bank_zq_tensor,
    bank_case_names,
    bank_paths,
    ly1,
    ly2,
    lx1,
    lx2,
    top_k=5,
    chunk_size=256
):
    """
    Fast context-matched prototype search using chunked batch computation.

    target_zq: [1,64,64,64] on GPU
    bank_zq_tensor: [N,64,64,64] on CPU
    """

    cy1, cy2, cx1, cx2 = get_context_box(
        ly1,
        ly2,
        lx1,
        lx2,
        context_pad
    )

    target_ctx = target_zq[:, :, cy1:cy2, cx1:cx2]

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

    mse_list = []

    n_bank = bank_zq_tensor.shape[0]

    with torch.no_grad():
        for start in range(0, n_bank, chunk_size):
            end = min(start + chunk_size, n_bank)

            bank_chunk = bank_zq_tensor[start:end].to(
                target_zq.device,
                non_blocking=True
            )

            bank_ctx = bank_chunk[:, :, cy1:cy2, cx1:cx2]

            diff = (bank_ctx - target_ctx) ** 2
            diff = diff * ring_mask

            mse = diff.sum(dim=[1, 2, 3]) / (
                ring_mask.sum() * target_zq.shape[1] + 1e-8
            )

            mse_list.append(mse.detach().cpu())

    all_mse = torch.cat(mse_list, dim=0)

    vals, idxs = torch.topk(
        all_mse,
        k=top_k,
        largest=False
    )

    prototypes = []

    for rank in range(top_k):
        idx = int(idxs[rank].item())

        prototypes.append({
            "path": bank_paths[idx],
            "case_name": bank_case_names[idx],
            "zq": bank_zq_tensor[idx:idx+1],
            "context_mse": float(vals[rank].item()),
            "bank_index": idx,
        })

    return prototypes


def apply_single_prototype_replacement(
    z_q,
    ly1,
    ly2,
    lx1,
    lx2,
    proto_zq,
    alpha
):
    z_corr = z_q.clone()

    proto_zq = proto_zq.to(z_q.device)

    z_corr[:, :, ly1:ly2, lx1:lx2] = (
        (1.0 - alpha) * z_q[:, :, ly1:ly2, lx1:lx2]
        + alpha * proto_zq[:, :, ly1:ly2, lx1:lx2]
    )

    return z_corr


def apply_candidate_infos_replacement(
    z_q,
    candidate_infos,
    alpha,
    prototype_rank=0
):
    z_corr = z_q.clone()

    for info in candidate_infos:
        proto = info["prototypes"][prototype_rank]
        proto_zq = proto["zq"].to(z_q.device)

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
# Candidate reranking
# ============================================================

def evaluate_candidate_consistency(
    x,
    z_q,
    diff_original,
    candidate,
    prototypes
):
    """
    For one candidate:
    - use topK prototypes
    - replacement with alpha_for_rerank
    - calculate region gain, outside gain, stability, context mse
    """

    region_mask = torch.zeros_like(x)
    region_mask[:, :, candidate["y1"]:candidate["y2"], candidate["x1"]:candidate["x2"]] = 1.0

    ordinary_region_diff = (
        diff_original * region_mask
    ).sum().item() / (region_mask.sum().item() + 1e-8)

    ordinary_outside_diff = (
        diff_original * (1.0 - region_mask)
    ).sum().item() / ((1.0 - region_mask).sum().item() + 1e-8)

    region_gains = []
    outside_gains = []
    region_diffs = []
    outside_diffs = []
    whole_diffs = []
    context_mses = []

    ly1 = candidate["ly1"]
    ly2 = candidate["ly2"]
    lx1 = candidate["lx1"]
    lx2 = candidate["lx2"]

    with torch.no_grad():
        for proto in prototypes:
            z_corr = apply_single_prototype_replacement(
                z_q,
                ly1,
                ly2,
                lx1,
                lx2,
                proto["zq"],
                alpha=alpha_for_rerank
            )

            recon_corr = model.decoder(z_corr)

            diff_corr, whole_diff, region_diff, outside_diff = compute_scores(
                x,
                recon_corr,
                region_mask
            )

            region_gain = region_diff - ordinary_region_diff
            outside_gain = outside_diff - ordinary_outside_diff

            region_gains.append(region_gain)
            outside_gains.append(outside_gain)
            region_diffs.append(region_diff)
            outside_diffs.append(outside_diff)
            whole_diffs.append(whole_diff)
            context_mses.append(proto["context_mse"])

    region_gains_np = np.array(region_gains, dtype=np.float32)
    outside_gains_np = np.array(outside_gains, dtype=np.float32)
    context_mses_np = np.array(context_mses, dtype=np.float32)

    mean_region_gain = float(region_gains_np.mean())
    std_region_gain = float(region_gains_np.std())
    mean_outside_gain = float(outside_gains_np.mean())
    std_outside_gain = float(outside_gains_np.std())
    best_context_mse = float(context_mses_np.min())
    mean_context_mse = float(context_mses_np.mean())

    # alpha monotonicity: test alpha 0.5, 0.7, 1.0 using best prototype only
    best_proto = prototypes[0]
    alpha_region_gains = []

    with torch.no_grad():
        for alpha in alpha_list:
            z_corr = apply_single_prototype_replacement(
                z_q,
                ly1,
                ly2,
                lx1,
                lx2,
                best_proto["zq"],
                alpha=alpha
            )

            recon_corr = model.decoder(z_corr)

            _, _, region_diff, _ = compute_scores(
                x,
                recon_corr,
                region_mask
            )

            alpha_region_gains.append(region_diff - ordinary_region_diff)

    monotonic_count = 0
    if alpha_region_gains[1] >= alpha_region_gains[0]:
        monotonic_count += 1
    if alpha_region_gains[2] >= alpha_region_gains[1]:
        monotonic_count += 1

    monotonic_score = monotonic_count / 2.0

    final_score = (
        mean_region_gain
        - outside_weight * mean_outside_gain
        - prototype_std_weight * std_region_gain
        - context_mse_weight * best_context_mse
        + monotonic_weight * monotonic_score
    )

    result = {
        "mean_region_gain": mean_region_gain,
        "std_region_gain": std_region_gain,
        "mean_outside_gain": mean_outside_gain,
        "std_outside_gain": std_outside_gain,
        "best_context_mse": best_context_mse,
        "mean_context_mse": mean_context_mse,
        "monotonic_score": monotonic_score,
        "final_score": float(final_score),
        "ordinary_region_diff": ordinary_region_diff,
        "ordinary_outside_diff": ordinary_outside_diff,
        "alpha05_region_gain": float(alpha_region_gains[0]),
        "alpha07_region_gain": float(alpha_region_gains[1]),
        "alpha10_region_gain": float(alpha_region_gains[2]),
    }

    return result


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

print("Stacking normal bank z_q tensor on CPU...")
bank_zq_tensor = torch.cat(
    [item["zq"] for item in normal_bank],
    dim=0
).float()

bank_case_names = [item["case_name"] for item in normal_bank]
bank_paths = [item["path"] for item in normal_bank]

print("bank_zq_tensor:", bank_zq_tensor.shape)


# ============================================================
# Evaluate one DICOM
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
    # Step 1: coarse image-level candidate generation
    # --------------------------------------------------------
    coarse_candidates, response_map, lung_mask = image_candidate_search(
        model,
        x
    )

    print("Coarse candidates:", len(coarse_candidates))

    # --------------------------------------------------------
    # Step 2: prototype search + consistency reranking
    # --------------------------------------------------------
    candidate_infos = []

    for i, c in enumerate(coarse_candidates):
        ly1, ly2, lx1, lx2 = image_box_to_latent_box(
            c["y1"],
            c["y2"],
            c["x1"],
            c["x2"],
            pad_latent=1
        )

        prototypes = search_prototypes_fast(
            z_q.detach(),
            bank_zq_tensor,
            bank_case_names,
            bank_paths,
            ly1,
            ly2,
            lx1,
            lx2,
            top_k=prototype_top_k,
            chunk_size=prototype_search_chunk_size
        )

        info = dict(c)
        info["coarse_rank"] = i + 1
        info["ly1"] = ly1
        info["ly2"] = ly2
        info["lx1"] = lx1
        info["lx2"] = lx2
        info["prototypes"] = prototypes

        consistency = evaluate_candidate_consistency(
            x,
            z_q,
            diff_original,
            info,
            prototypes
        )

        info.update(consistency)

        candidate_infos.append(info)

        print(
            f"coarse {i+1:02d}: "
            f"img_score={info['score']:.6f}, "
            f"final={info['final_score']:.6f}, "
            f"region_gain={info['mean_region_gain']:.6f}, "
            f"outside_gain={info['mean_outside_gain']:.6f}, "
            f"std={info['std_region_gain']:.6f}, "
            f"ctx={info['best_context_mse']:.6f}, "
            f"y={info['y1']}:{info['y2']}, x={info['x1']}:{info['x2']}"
        )

    reranked_infos = sorted(
        candidate_infos,
        key=lambda c: c["final_score"],
        reverse=True
    )

    print("Reranked top candidates:")
    for i, info in enumerate(reranked_infos[:5]):
        print(
            f"rank {i+1}: "
            f"final={info['final_score']:.6f}, "
            f"coarse_rank={info['coarse_rank']}, "
            f"region_gain={info['mean_region_gain']:.6f}, "
            f"outside_gain={info['mean_outside_gain']:.6f}, "
            f"ctx={info['best_context_mse']:.6f}, "
            f"y={info['y1']}:{info['y2']}, x={info['x1']}:{info['x2']}"
        )

    records = []

    # --------------------------------------------------------
    # Step 3: calculate top1/top3/top5 union scores after reranking
    # --------------------------------------------------------
    for n_candidate in [1, 3, 5]:
        n_candidate = min(n_candidate, len(reranked_infos))

        selected_infos = reranked_infos[:n_candidate]
        selected_candidates = selected_infos

        union_mask = make_union_mask(x, selected_candidates)

        ordinary_region_diff = (
            diff_original * union_mask
        ).sum().item() / (union_mask.sum().item() + 1e-8)

        ordinary_outside_diff = (
            diff_original * (1.0 - union_mask)
        ).sum().item() / ((1.0 - union_mask).sum().item() + 1e-8)

        for alpha in alpha_list:
            with torch.no_grad():
                z_corr = apply_candidate_infos_replacement(
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
    # Candidate detail records
    # --------------------------------------------------------
    for final_rank, info in enumerate(reranked_infos):
        for p_rank, proto in enumerate(info["prototypes"]):
            records.append({
                "label": label,
                "case_name": case_name,
                "path": dicom_path,

                "record_type": "candidate_prototype",

                "final_rank": final_rank + 1,
                "coarse_rank": info["coarse_rank"],

                "coarse_score": info["score"],
                "final_score": info["final_score"],

                "candidate_y1": info["y1"],
                "candidate_y2": info["y2"],
                "candidate_x1": info["x1"],
                "candidate_x2": info["x2"],
                "candidate_lung_ratio": info["lung_ratio"],

                "latent_y1": info["ly1"],
                "latent_y2": info["ly2"],
                "latent_x1": info["lx1"],
                "latent_x2": info["lx2"],

                "mean_region_gain": info["mean_region_gain"],
                "std_region_gain": info["std_region_gain"],
                "mean_outside_gain": info["mean_outside_gain"],
                "std_outside_gain": info["std_outside_gain"],
                "best_context_mse": info["best_context_mse"],
                "mean_context_mse": info["mean_context_mse"],
                "monotonic_score": info["monotonic_score"],

                "alpha05_region_gain": info["alpha05_region_gain"],
                "alpha07_region_gain": info["alpha07_region_gain"],
                "alpha10_region_gain": info["alpha10_region_gain"],

                "prototype_rank": p_rank + 1,
                "prototype_case": proto["case_name"],
                "prototype_path": proto["path"],
                "prototype_bank_index": proto["bank_index"],
                "context_mse": proto["context_mse"],
            })

    # --------------------------------------------------------
    # Save visualization
    # --------------------------------------------------------
    should_save = True

    if max_save_images is not None and case_idx >= max_save_images:
        should_save = False

    if should_save:
        masks = []
        recons = []
        diffs = []

        for n_candidate in [1, 3, 5]:
            n_candidate = min(n_candidate, len(reranked_infos))

            selected_infos = reranked_infos[:n_candidate]
            union_mask = make_union_mask(x, selected_infos)

            with torch.no_grad():
                z_corr = apply_candidate_infos_replacement(
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

            top1_overlay,                             # 6 rerank top1 overlay
            masks[0],                                 # 7 rerank top1 mask
            recons[0],                                # 8 rerank top1 prototype recon
            diffs[0],                                 # 9 rerank top1 prototype diff

            top3_overlay,                             # 10 rerank top3 overlay
            masks[1],                                 # 11 rerank top3 mask
            recons[1],                                # 12 rerank top3 prototype recon
            diffs[1],                                 # 13 rerank top3 prototype diff

            masks[2],                                 # 14 rerank top5 mask
            recons[2],                                # 15 rerank top5 prototype recon
            diffs[2],                                 # 16 rerank top5 prototype diff

            normalize_map(q_map).cpu(),               # 17 qdistance
        ]

        grid = vutils.make_grid(
            torch.cat(grid_items, dim=0),
            nrow=17,
            normalize=True
        )

        save_img_path = os.path.join(
            case_save_dir,
            f"{case_name}_stage8_5_rerank.png"
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
    "stage8_5_consistency_rerank_scores.csv"
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
    "stage8_5_summary_by_label_topn_alpha.csv"
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
    "stage8_5_detection_summary_normal_threshold.csv"
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
print("6 rerank top1 overlay")
print("7 rerank top1 mask")
print("8 rerank top1 prototype recon")
print("9 rerank top1 prototype diff")
print("10 rerank top3 overlay")
print("11 rerank top3 mask")
print("12 rerank top3 prototype recon")
print("13 rerank top3 prototype diff")
print("14 rerank top5 mask")
print("15 rerank top5 prototype recon")
print("16 rerank top5 prototype diff")
print("17 qdistance")