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

# normal train data for prototype bank
normal_train_dir = "/app/data/train"

# Stage1 full VQ-VAE
model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage1_full_final.pth"

save_root = "/app/results/eval_stage9_rerank_normal_abnormal"
os.makedirs(save_root, exist_ok=True)

resize = 256

# ------------------------------------------------------------
# Coarse image-level candidate generation
# ------------------------------------------------------------
image_block_sizes = [32, 40, 48]
image_stride = 16
fill_value = 0.08
min_lung_ratio = 0.25

coarse_top_n = 30
final_top_n = 5
candidate_nms_iou = 0.30

score_mode = "topk"
topk_ratio = 0.10

# ------------------------------------------------------------
# Prototype replacement
# ------------------------------------------------------------
max_normal_samples = 500
prototype_top_k = 5

alpha_list = [0.5, 0.7, 1.0]
vis_alpha = 0.7

context_pad = 4

# ------------------------------------------------------------
# Reranking weights
# ------------------------------------------------------------
outside_weight = 30.0
prototype_std_weight = 0.3
complexity_weight = 0.10
concentration_weight = 0.02
monotonic_weight = 0.5

# None means save all images
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

    # lung window [-1000, 400]
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


def topk_mean(x, ratio=0.10):
    flat = x.reshape(-1)
    k = max(1, int(flat.numel() * ratio))
    vals, _ = torch.topk(flat, k)
    return vals.mean()


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
# Lung mask
# ============================================================

def make_lung_mask(x):
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


# ============================================================
# Normal structure complexity
# ============================================================

def compute_gradient_map(x):
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)

    gx[:, :, :, 1:] = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    gy[:, :, 1:, :] = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])

    grad = gx + gy

    return grad


def compute_local_complexity(x, y1, y2, x1, x2):
    patch = x[:, :, y1:y2, x1:x2]

    grad_map = compute_gradient_map(x)
    grad_patch = grad_map[:, :, y1:y2, x1:x2]

    grad_mean = grad_patch.mean().item()
    var_score = patch.var().item()

    complexity = grad_mean + var_score

    return complexity, grad_mean, var_score


def compute_response_concentration(diff, y1, y2, x1, x2):
    block = diff[:, :, y1:y2, x1:x2]

    mean_val = block.mean()
    topk_val = topk_mean(block, ratio=topk_ratio)

    concentration = (topk_val / (mean_val + 1e-8)).item()

    return concentration


# ============================================================
# Candidate generation
# ============================================================

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


def nms_candidates(candidates, top_n=30, iou_threshold=0.30, score_key="score"):
    candidates = sorted(candidates, key=lambda c: c[score_key], reverse=True)

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
    粗候选生成。
    image-level mask response 只用于取得 top30，不作为最终排序。
    """

    lung_mask = make_lung_mask(x)

    response_map = torch.zeros_like(x)
    response_count = torch.zeros_like(x)

    candidates = []

    with torch.no_grad():
        for block_size in image_block_sizes:
            for y1 in range(0, resize - block_size + 1, image_stride):
                for x1 in range(0, resize - block_size + 1, image_stride):

                    y2 = y1 + block_size
                    x2 = x1 + block_size

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

                    complexity, grad_mean, var_score = compute_local_complexity(
                        x,
                        y1,
                        y2,
                        x1,
                        x2
                    )

                    concentration = compute_response_concentration(
                        diff,
                        y1,
                        y2,
                        x1,
                        x2
                    )

                    candidates.append({
                        "score": score,
                        "image_score": score,
                        "block_size": block_size,
                        "y1": y1,
                        "y2": y2,
                        "x1": x1,
                        "x2": x2,
                        "lung_ratio": lung_ratio,
                        "complexity": complexity,
                        "grad_mean": grad_mean,
                        "var_score": var_score,
                        "image_concentration": concentration,
                    })

                    response_map[:, :, y1:y2, x1:x2] += score
                    response_count[:, :, y1:y2, x1:x2] += 1.0

    if len(candidates) == 0:
        print("Warning: no lung-constrained candidates. Fallback to all blocks.")

        with torch.no_grad():
            for block_size in image_block_sizes:
                for y1 in range(0, resize - block_size + 1, image_stride):
                    for x1 in range(0, resize - block_size + 1, image_stride):

                        y2 = y1 + block_size
                        x2 = x1 + block_size

                        x_masked = x.clone()
                        x_masked[:, :, y1:y2, x1:x2] = fill_value

                        recon_masked, _, _, _ = extract_zq_recon_qdist(
                            model,
                            x_masked
                        )

                        diff = torch.abs(x - recon_masked)

                        block_diff = diff[:, :, y1:y2, x1:x2]
                        score = compute_block_score(block_diff)

                        complexity, grad_mean, var_score = compute_local_complexity(
                            x,
                            y1,
                            y2,
                            x1,
                            x2
                        )

                        concentration = compute_response_concentration(
                            diff,
                            y1,
                            y2,
                            x1,
                            x2
                        )

                        candidates.append({
                            "score": score,
                            "image_score": score,
                            "block_size": block_size,
                            "y1": y1,
                            "y2": y2,
                            "x1": x1,
                            "x2": x2,
                            "lung_ratio": 0.0,
                            "complexity": complexity,
                            "grad_mean": grad_mean,
                            "var_score": var_score,
                            "image_concentration": concentration,
                        })

                        response_map[:, :, y1:y2, x1:x2] += score
                        response_count[:, :, y1:y2, x1:x2] += 1.0

    response_map = response_map / (response_count + 1e-8)

    selected = nms_candidates(
        candidates,
        top_n=coarse_top_n,
        iou_threshold=candidate_nms_iou,
        score_key="score"
    )

    return selected, response_map, lung_mask


# ============================================================
# Coordinate conversion
# ============================================================

def image_box_to_latent_box(y1, y2, x1, x2, pad_latent=1):
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
# Prototype bank and replacement
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
# Reranking
# ============================================================

def evaluate_candidate_by_prototype_response(
    x,
    z_q,
    diff_original,
    candidate,
    normal_bank
):
    y1, y2, x1, x2 = candidate["y1"], candidate["y2"], candidate["x1"], candidate["x2"]

    ly1, ly2, lx1, lx2 = image_box_to_latent_box(
        y1,
        y2,
        x1,
        x2,
        pad_latent=1
    )

    candidate_mask = torch.zeros_like(x)
    candidate_mask[:, :, y1:y2, x1:x2] = 1.0

    ordinary_region_diff = (
        diff_original * candidate_mask
    ).sum().item() / (candidate_mask.sum().item() + 1e-8)

    ordinary_outside_diff = (
        diff_original * (1.0 - candidate_mask)
    ).sum().item() / ((1.0 - candidate_mask).sum().item() + 1e-8)

    prototypes = search_prototypes(
        z_q.detach(),
        normal_bank,
        ly1,
        ly2,
        lx1,
        lx2
    )

    region_gain_list = []
    outside_gain_list = []
    concentration_list = []

    alpha_gain_dict = {alpha: [] for alpha in alpha_list}

    for proto in prototypes:
        proto_zq = proto["zq"].to(device)

        for alpha in alpha_list:
            with torch.no_grad():
                z_corr = replace_with_prototype(
                    z_q,
                    proto_zq,
                    ly1,
                    ly2,
                    lx1,
                    lx2,
                    alpha
                )

                recon_corr = model.decoder(z_corr)

                diff_corr, whole_diff, region_diff, outside_diff = compute_scores(
                    x,
                    recon_corr,
                    candidate_mask
                )

            region_gain = region_diff - ordinary_region_diff
            outside_gain = outside_diff - ordinary_outside_diff

            concentration = compute_response_concentration(
                diff_corr,
                y1,
                y2,
                x1,
                x2
            )

            region_gain_list.append(region_gain)
            outside_gain_list.append(outside_gain)
            concentration_list.append(concentration)
            alpha_gain_dict[alpha].append(region_gain)

    mean_region_gain = float(np.mean(region_gain_list))
    mean_outside_gain = float(np.mean(outside_gain_list))
    std_region_gain = float(np.std(region_gain_list))
    mean_concentration = float(np.mean(concentration_list))

    gain_05 = float(np.mean(alpha_gain_dict[0.5]))
    gain_07 = float(np.mean(alpha_gain_dict[0.7]))
    gain_10 = float(np.mean(alpha_gain_dict[1.0]))

    monotonic_gain = gain_10 - gain_05
    is_monotonic = int(gain_05 < gain_07 < gain_10)

    rerank_score = (
        mean_region_gain
        - outside_weight * mean_outside_gain
        - prototype_std_weight * std_region_gain
        - complexity_weight * candidate["complexity"]
        + concentration_weight * mean_concentration
        + monotonic_weight * monotonic_gain
    )

    out = dict(candidate)

    out.update({
        "ly1": ly1,
        "ly2": ly2,
        "lx1": lx1,
        "lx2": lx2,

        "prototypes": prototypes,

        "ordinary_region_diff": ordinary_region_diff,
        "ordinary_outside_diff": ordinary_outside_diff,

        "mean_region_gain": mean_region_gain,
        "mean_outside_gain": mean_outside_gain,
        "std_region_gain": std_region_gain,
        "mean_concentration": mean_concentration,

        "gain_alpha_05": gain_05,
        "gain_alpha_07": gain_07,
        "gain_alpha_10": gain_10,

        "monotonic_gain": monotonic_gain,
        "is_monotonic": is_monotonic,

        "rerank_score": rerank_score,
    })

    return out


def apply_reranked_candidate_replacement(z_q, candidate_infos, alpha=0.7):
    z_corr = z_q.clone()

    for info in candidate_infos:
        proto = info["prototypes"][0]
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


# ============================================================
# Build normal prototype bank
# ============================================================

normal_train_paths = collect_dicom_paths(
    normal_train_dir,
    max_samples=max_normal_samples
)

normal_bank = build_normal_zq_bank(
    model,
    normal_train_paths
)


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
    # Step 1: coarse candidates
    # --------------------------------------------------------
    coarse_candidates, response_map, lung_mask = image_candidate_search(
        model,
        x
    )

    print(f"Coarse candidates: {len(coarse_candidates)}")

    # --------------------------------------------------------
    # Step 2: prototype-response reranking
    # --------------------------------------------------------
    reranked_infos = []

    for i, c in enumerate(coarse_candidates):
        info = evaluate_candidate_by_prototype_response(
            x,
            z_q,
            diff_original,
            c,
            normal_bank
        )

        info["coarse_rank"] = i + 1
        reranked_infos.append(info)

        print(
            f"candidate {i+1:02d}: "
            f"img_score={info['image_score']:.6f}, "
            f"rerank={info['rerank_score']:.6f}, "
            f"region_gain={info['mean_region_gain']:.6f}, "
            f"outside_gain={info['mean_outside_gain']:.6f}, "
            f"complexity={info['complexity']:.6f}, "
            f"conc={info['mean_concentration']:.3f}, "
            f"mono={info['is_monotonic']}, "
            f"box=({info['y1']}:{info['y2']},{info['x1']}:{info['x2']}), "
            f"size={info['block_size']}"
        )

    reranked_infos = sorted(
        reranked_infos,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    reranked_infos = nms_candidates(
        reranked_infos,
        top_n=final_top_n,
        iou_threshold=candidate_nms_iou,
        score_key="rerank_score"
    )

    print("Final reranked candidates:")
    for i, info in enumerate(reranked_infos):
        print(
            f"rerank top{i+1}: "
            f"score={info['rerank_score']:.6f}, "
            f"coarse_rank={info['coarse_rank']}, "
            f"y={info['y1']}:{info['y2']}, "
            f"x={info['x1']}:{info['x2']}, "
            f"size={info['block_size']}"
        )

    records = []

    # --------------------------------------------------------
    # Step 3: reconstruct top1/top3/top5
    # --------------------------------------------------------
    for n_candidate in [1, 3, 5]:
        n_candidate = min(n_candidate, len(reranked_infos))

        selected_infos = reranked_infos[:n_candidate]
        union_mask = make_union_mask(x, selected_infos)

        ordinary_region_diff = (
            diff_original * union_mask
        ).sum().item() / (union_mask.sum().item() + 1e-8)

        ordinary_outside_diff = (
            diff_original * (1.0 - union_mask)
        ).sum().item() / ((1.0 - union_mask).sum().item() + 1e-8)

        for alpha in alpha_list:
            with torch.no_grad():
                z_corr = apply_reranked_candidate_replacement(
                    z_q,
                    selected_infos,
                    alpha=alpha
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
    for rank, info in enumerate(reranked_infos):
        proto0 = info["prototypes"][0]

        records.append({
            "label": label,
            "case_name": case_name,
            "path": dicom_path,

            "record_type": "candidate_detail",
            "rerank_rank": rank + 1,
            "coarse_rank": info["coarse_rank"],

            "rerank_score": info["rerank_score"],
            "image_score": info["image_score"],

            "y1": info["y1"],
            "y2": info["y2"],
            "x1": info["x1"],
            "x2": info["x2"],

            "ly1": info["ly1"],
            "ly2": info["ly2"],
            "lx1": info["lx1"],
            "lx2": info["lx2"],

            "block_size": info["block_size"],
            "lung_ratio": info["lung_ratio"],

            "complexity": info["complexity"],
            "grad_mean": info["grad_mean"],
            "var_score": info["var_score"],

            "mean_region_gain": info["mean_region_gain"],
            "mean_outside_gain": info["mean_outside_gain"],
            "std_region_gain": info["std_region_gain"],
            "mean_concentration": info["mean_concentration"],

            "gain_alpha_05": info["gain_alpha_05"],
            "gain_alpha_07": info["gain_alpha_07"],
            "gain_alpha_10": info["gain_alpha_10"],
            "monotonic_gain": info["monotonic_gain"],
            "is_monotonic": info["is_monotonic"],

            "prototype_case": proto0["case_name"],
            "prototype_path": proto0["path"],
            "context_mse": proto0["context_mse"],
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
                z_corr = apply_reranked_candidate_replacement(
                    z_q,
                    selected_infos,
                    alpha=vis_alpha
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
            recons[0],                                # 8 rerank top1 recon
            diffs[0],                                 # 9 rerank top1 diff

            top3_overlay,                             # 10 rerank top3 overlay
            masks[1],                                 # 11 rerank top3 mask
            recons[1],                                # 12 rerank top3 recon
            diffs[1],                                 # 13 rerank top3 diff

            masks[2],                                 # 14 rerank top5 mask
            recons[2],                                # 15 rerank top5 recon
            diffs[2],                                 # 16 rerank top5 diff

            normalize_map(q_map).cpu(),               # 17 qdistance
        ]

        grid = vutils.make_grid(
            torch.cat(grid_items, dim=0),
            nrow=17,
            normalize=True
        )

        save_img_path = os.path.join(
            case_save_dir,
            f"{case_name}_stage9_rerank_normal_abnormal.png"
        )

        vutils.save_image(grid, save_img_path)

        print("Saved image:", save_img_path)

    return records


# ============================================================
# Run normal + abnormal
# ============================================================

normal_test_paths = collect_dicom_paths(normal_test_dir)
abnormal_paths = collect_dicom_paths(abnormal_dir)

print()
print("Normal test cases:", len(normal_test_paths))
print("Abnormal cases:", len(abnormal_paths))

all_records = []

for idx, path in enumerate(normal_test_paths):
    case_records = evaluate_case(idx, path, label="normal")
    all_records.extend(case_records)

for idx, path in enumerate(abnormal_paths):
    case_records = evaluate_case(idx, path, label="abnormal")
    all_records.extend(case_records)


# ============================================================
# Save CSV
# ============================================================

df = pd.DataFrame(all_records)

csv_path = os.path.join(
    save_root,
    "stage9_rerank_normal_abnormal_scores.csv"
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
    "stage9_summary_by_label_topn_alpha.csv"
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
    "stage9_detection_summary_normal_threshold.csv"
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