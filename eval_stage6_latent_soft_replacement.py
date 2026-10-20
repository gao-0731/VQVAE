import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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

normal_data_dir = "/app/data/temp/normal"
abnormal_data_dir = "/app/data/temp/abnormal"

# Stage1 full VQ-VAE
model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage1_full_final.pth"

# Stage1 上计算出来的 z_q center/std
stats_path = "/app/results/zq_stats_stage1_full.pth"

save_root = "/app/results/eval_stage6_latent_soft_replacement"

resize = 256
batch_size = 1
num_workers = 2

# latent map is 64x64, image is 256x256
latent_size = 64
image_size = 256
scale = image_size // latent_size   # 4

# latent-level sliding block
latent_block_size = 10
latent_stride = 4

# soft replacement strength
alpha_list = [0.3, 0.5, 0.7, 1.0]

# final visualization reconstruction uses this alpha
final_alpha = 0.7

# score inside block uses top-k pixels
topk_ratio = 0.10

# candidate top-N
top_n = 5

# lung constraint on corresponding image region
use_lung_constraint = True
min_lung_ratio = 0.20

# response score weights
monotonic_weight = 0.5
outside_penalty_weight = 0.2

# None means save all images
max_save_images = None


# ============================================================
# Model Parameters
# ============================================================

n_hiddens = 128
n_residual_hiddens = 64
n_residual_layers = 1

embedding_dim = 64
n_embeddings = 64
beta = 0.25


# ============================================================
# Dataset
# ============================================================

class DicomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, label):
        self.paths = []
        self.label = label

        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(".dcm"):
                    self.paths.append(os.path.join(root, f))

        self.paths = sorted(self.paths)
        print(f"[{label}] Found {len(self.paths)} DICOM files in {root_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        ds = pydicom.dcmread(path)
        image = ds.pixel_array.astype(np.float32)

        # HU conversion
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        image = image * slope + intercept

        # lung window [-1000, 400]
        window_min = -1000.0
        window_max = 400.0

        image = np.clip(image, window_min, window_max)
        image = (image - window_min) / (window_max - window_min)

        # resize to 256 x 256
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = image.resize((resize, resize), Image.BILINEAR)
        image = np.array(image).astype(np.float32) / 255.0

        image = torch.tensor(image).unsqueeze(0)  # [1,256,256]

        case_name = os.path.basename(path).replace(".dcm", "")

        return image, path, case_name


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
# Load z_q stats
# ============================================================

stats = torch.load(stats_path, map_location=device)

center_map = stats["center_map"].to(device)
std_map = stats["std_map"].to(device)

print("Loaded stats:", stats_path)
print("center_map:", center_map.shape)
print("std_map:", std_map.shape)


# ============================================================
# Utility functions
# ============================================================

def normalize_map(x):
    x = x.detach()
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def topk_mean(x, ratio=0.10):
    flat = x.reshape(-1)
    k = max(1, int(flat.numel() * ratio))
    vals, _ = torch.topk(flat, k)
    return vals.mean()


def make_lung_mask(x):
    """
    x: [1,1,256,256]
    return: [1,1,256,256]
    """

    lung = ((x > 0.02) & (x < 0.45)).float()

    # slight dilation
    lung = F.max_pool2d(
        lung,
        kernel_size=5,
        stride=1,
        padding=2
    )

    # slight erosion
    lung = -F.max_pool2d(
        -lung,
        kernel_size=5,
        stride=1,
        padding=2
    )

    lung = (lung > 0.5).float()

    return lung


def extract_zq_recon_qdist(model, x):
    """
    x: [1,1,256,256]

    return:
        recon: [1,1,256,256]
        z_q: [1,64,64,64]
        q_distance_map: [1,64,64]
        perplexity
    """

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


def latent_to_image_box(ly1, ly2, lx1, lx2):
    """
    latent coordinates -> image coordinates
    """

    y1 = ly1 * scale
    y2 = ly2 * scale
    x1 = lx1 * scale
    x2 = lx2 * scale

    return y1, y2, x1, x2


def make_image_mask_from_latent_box(x, ly1, ly2, lx1, lx2):
    y1, y2, x1, x2 = latent_to_image_box(ly1, ly2, lx1, lx2)

    mask = torch.zeros_like(x)
    mask[:, :, y1:y2, x1:x2] = 1.0

    return mask


def soft_replace_latent_block(z_q, center_map, ly1, ly2, lx1, lx2, alpha):
    """
    z_q_corrected = (1-alpha)*z_q + alpha*center_map
    only inside selected block
    """

    z_corr = z_q.clone()

    z_corr[:, :, ly1:ly2, lx1:lx2] = (
        (1.0 - alpha) * z_q[:, :, ly1:ly2, lx1:lx2]
        + alpha * center_map[:, :, ly1:ly2, lx1:lx2]
    )

    return z_corr


def decode_corrected_latent(model, z_corr):
    return model.decoder(z_corr)


def compute_candidate_response(model, x, z_q, center_map, lung_mask, ly1, ly2, lx1, lx2):
    """
    For one latent candidate block, test multiple alpha values.
    Score is based on:
    - inside response mean across alphas
    - monotonic increase with alpha
    - outside artifact penalty
    """

    y1, y2, x1, x2 = latent_to_image_box(ly1, ly2, lx1, lx2)

    image_mask = torch.zeros_like(x)
    image_mask[:, :, y1:y2, x1:x2] = 1.0

    if use_lung_constraint:
        lung_ratio = (lung_mask * image_mask).sum().item() / (image_mask.sum().item() + 1e-8)
        if lung_ratio < min_lung_ratio:
            return None
    else:
        lung_ratio = 1.0

    inside_scores = []
    outside_scores = []
    alpha_diffs = []

    for alpha in alpha_list:
        z_corr = soft_replace_latent_block(
            z_q,
            center_map,
            ly1,
            ly2,
            lx1,
            lx2,
            alpha
        )

        recon_corr = decode_corrected_latent(model, z_corr)
        diff = torch.abs(x - recon_corr)

        inside_diff = diff[:, :, y1:y2, x1:x2]
        inside_score = topk_mean(inside_diff, ratio=topk_ratio)

        outside_mask = 1.0 - image_mask
        outside_score = (diff * outside_mask).sum() / (outside_mask.sum() + 1e-8)

        inside_scores.append(float(inside_score.item()))
        outside_scores.append(float(outside_score.item()))
        alpha_diffs.append(diff)

    response_mean = float(np.mean(inside_scores))
    outside_mean = float(np.mean(outside_scores))

    # monotonicity: encourage response to increase when alpha gets stronger
    monotonic_gain = inside_scores[-1] - inside_scores[0]

    score = (
        response_mean
        + monotonic_weight * monotonic_gain
        - outside_penalty_weight * outside_mean
    )

    candidate = {
        "score": score,
        "response_mean": response_mean,
        "monotonic_gain": monotonic_gain,
        "outside_mean": outside_mean,
        "lung_ratio": lung_ratio,

        "ly1": ly1,
        "ly2": ly2,
        "lx1": lx1,
        "lx2": lx2,

        "y1": y1,
        "y2": y2,
        "x1": x1,
        "x2": x2,
    }

    for i, alpha in enumerate(alpha_list):
        candidate[f"inside_alpha_{alpha}"] = inside_scores[i]
        candidate[f"outside_alpha_{alpha}"] = outside_scores[i]

    return candidate


def candidate_iou(c1, c2):
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
            if candidate_iou(c, s) > iou_threshold:
                keep = False
                break

        if keep:
            selected.append(c)

        if len(selected) >= top_n:
            break

    return selected


def make_union_image_mask(x, candidates):
    mask = torch.zeros_like(x)

    for c in candidates:
        mask[:, :, c["y1"]:c["y2"], c["x1"]:c["x2"]] = 1.0

    return mask


def apply_latent_replacement_for_candidates(z_q, center_map, candidates, alpha):
    z_corr = z_q.clone()

    for c in candidates:
        ly1, ly2 = c["ly1"], c["ly2"]
        lx1, lx2 = c["lx1"], c["lx2"]

        z_corr[:, :, ly1:ly2, lx1:lx2] = (
            (1.0 - alpha) * z_q[:, :, ly1:ly2, lx1:lx2]
            + alpha * center_map[:, :, ly1:ly2, lx1:lx2]
        )

    return z_corr


def reconstruct_with_candidates(model, x, z_q, center_map, candidates, n_candidate, alpha):
    selected = candidates[:n_candidate]

    union_mask = make_union_image_mask(x, selected)

    z_corr = apply_latent_replacement_for_candidates(
        z_q,
        center_map,
        selected,
        alpha
    )

    recon = model.decoder(z_corr)
    diff = torch.abs(x - recon)

    region_diff = (
        diff * union_mask
    ).sum().item() / (union_mask.sum().item() + 1e-8)

    return union_mask, recon, diff, region_diff


def latent_replacement_search(model, x, z_q, center_map):
    """
    sliding search in latent space
    """

    lung_mask = make_lung_mask(x)

    response_map = torch.zeros_like(x)
    response_count = torch.zeros_like(x)

    candidates = []

    total_blocks = 0
    valid_blocks = 0

    with torch.no_grad():
        for ly1 in range(0, latent_size - latent_block_size + 1, latent_stride):
            for lx1 in range(0, latent_size - latent_block_size + 1, latent_stride):

                ly2 = ly1 + latent_block_size
                lx2 = lx1 + latent_block_size

                total_blocks += 1

                candidate = compute_candidate_response(
                    model,
                    x,
                    z_q,
                    center_map,
                    lung_mask,
                    ly1,
                    ly2,
                    lx1,
                    lx2
                )

                if candidate is None:
                    continue

                valid_blocks += 1
                candidates.append(candidate)

                y1, y2, x1, x2 = candidate["y1"], candidate["y2"], candidate["x1"], candidate["x2"]

                response_map[:, :, y1:y2, x1:x2] += candidate["score"]
                response_count[:, :, y1:y2, x1:x2] += 1.0

    if valid_blocks == 0:
        print("Warning: no valid lung-constrained latent blocks. Fallback to all latent blocks.")

        candidates = []
        response_map = torch.zeros_like(x)
        response_count = torch.zeros_like(x)

        with torch.no_grad():
            for ly1 in range(0, latent_size - latent_block_size + 1, latent_stride):
                for lx1 in range(0, latent_size - latent_block_size + 1, latent_stride):

                    ly2 = ly1 + latent_block_size
                    lx2 = lx1 + latent_block_size

                    candidate = compute_candidate_response(
                        model,
                        x,
                        z_q,
                        center_map,
                        torch.ones_like(lung_mask),
                        ly1,
                        ly2,
                        lx1,
                        lx2
                    )

                    if candidate is None:
                        continue

                    candidates.append(candidate)

                    y1, y2, x1, x2 = candidate["y1"], candidate["y2"], candidate["x1"], candidate["x2"]

                    response_map[:, :, y1:y2, x1:x2] += candidate["score"]
                    response_count[:, :, y1:y2, x1:x2] += 1.0

        valid_blocks = len(candidates)

    response_map = response_map / (response_count + 1e-8)

    selected_candidates = nms_candidates(
        candidates,
        top_n=top_n,
        iou_threshold=0.30
    )

    return selected_candidates, response_map, lung_mask, valid_blocks, total_blocks


# ============================================================
# Evaluate dataset
# ============================================================

def evaluate_dataset(data_dir, label):
    dataset = DicomDataset(data_dir, label)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    save_dir = os.path.join(save_root, label)
    os.makedirs(save_dir, exist_ok=True)

    records = []

    with torch.no_grad():
        for idx, (x, path, case_name) in enumerate(loader):
            x = x.to(device)

            case_name = case_name[0]
            path = path[0]

            # ordinary reconstruction
            recon_original, z_q, q_distance_map, perplexity = extract_zq_recon_qdist(
                model,
                x
            )

            diff_original = torch.abs(x - recon_original)

            # latent response search
            (
                candidates,
                response_map,
                lung_mask,
                valid_blocks,
                total_blocks
            ) = latent_replacement_search(
                model,
                x,
                z_q,
                center_map
            )

            if len(candidates) == 0:
                print(f"Warning: no candidates found for {case_name}")
                continue

            top1_n = min(1, len(candidates))
            top3_n = min(3, len(candidates))
            top5_n = min(5, len(candidates))

            mask_top1, recon_top1, diff_top1, region_diff_top1 = reconstruct_with_candidates(
                model,
                x,
                z_q,
                center_map,
                candidates,
                top1_n,
                final_alpha
            )

            mask_top3, recon_top3, diff_top3, region_diff_top3 = reconstruct_with_candidates(
                model,
                x,
                z_q,
                center_map,
                candidates,
                top3_n,
                final_alpha
            )

            mask_top5, recon_top5, diff_top5, region_diff_top5 = reconstruct_with_candidates(
                model,
                x,
                z_q,
                center_map,
                candidates,
                top5_n,
                final_alpha
            )

            # qdistance map
            q_map = q_distance_map.unsqueeze(1)
            q_map = F.interpolate(
                q_map,
                size=(256, 256),
                mode="bilinear",
                align_corners=False
            )

            # scores
            ordinary_diff_score = diff_original.mean().item()

            corrected_diff_top1 = diff_top1.mean().item()
            corrected_diff_top3 = diff_top3.mean().item()
            corrected_diff_top5 = diff_top5.mean().item()

            response_mean = response_map.mean().item()
            response_max = response_map.max().item()

            qdist_mean = q_map.mean().item()
            qdist_max = q_map.max().item()

            c1 = candidates[0]
            c3_score = np.mean([c["score"] for c in candidates[:top3_n]])
            c5_score = np.mean([c["score"] for c in candidates[:top5_n]])

            # save image
            should_save = True
            if max_save_images is not None and idx >= max_save_images:
                should_save = False

            image_save_path = ""

            if should_save:
                grid = vutils.make_grid(
                    torch.cat([
                        x.cpu(),                                  # 1 input
                        recon_original.cpu(),                     # 2 ordinary recon
                        normalize_map(diff_original).cpu(),       # 3 ordinary diff
                        lung_mask.cpu(),                          # 4 lung mask
                        normalize_map(response_map).cpu(),        # 5 latent response map

                        mask_top1.cpu(),                          # 6 top1 latent mask image region
                        recon_top1.cpu(),                         # 7 top1 corrected recon
                        normalize_map(diff_top1).cpu(),           # 8 top1 corrected diff

                        mask_top3.cpu(),                          # 9 top3 union mask
                        recon_top3.cpu(),                         # 10 top3 corrected recon
                        normalize_map(diff_top3).cpu(),           # 11 top3 corrected diff

                        mask_top5.cpu(),                          # 12 top5 union mask
                        recon_top5.cpu(),                         # 13 top5 corrected recon
                        normalize_map(diff_top5).cpu(),           # 14 top5 corrected diff

                        normalize_map(q_map).cpu(),               # 15 qdistance original
                    ], dim=0),
                    nrow=15,
                    normalize=True
                )

                image_save_path = os.path.join(
                    save_dir,
                    f"{idx:04d}_{case_name}.png"
                )

                vutils.save_image(grid, image_save_path)

            record = {
                "label": label,
                "case_name": case_name,
                "path": path,

                "valid_blocks": valid_blocks,
                "total_blocks": total_blocks,

                "ordinary_diff": ordinary_diff_score,

                "corrected_diff_top1": corrected_diff_top1,
                "corrected_diff_top3": corrected_diff_top3,
                "corrected_diff_top5": corrected_diff_top5,

                "corrected_region_diff_top1": region_diff_top1,
                "corrected_region_diff_top3": region_diff_top3,
                "corrected_region_diff_top5": region_diff_top5,

                "best_response_score": c1["score"],
                "top3_response_score_mean": c3_score,
                "top5_response_score_mean": c5_score,

                "response_mean": response_mean,
                "response_max": response_max,

                "qdist_mean": qdist_mean,
                "qdist_max": qdist_max,

                "perplexity_original": float(perplexity.item()),

                "image_path": image_save_path,
            }

            for i in range(top_n):
                if i < len(candidates):
                    c = candidates[i]
                    record[f"top{i+1}_score"] = c["score"]
                    record[f"top{i+1}_response_mean"] = c["response_mean"]
                    record[f"top{i+1}_monotonic_gain"] = c["monotonic_gain"]
                    record[f"top{i+1}_outside_mean"] = c["outside_mean"]
                    record[f"top{i+1}_lung_ratio"] = c["lung_ratio"]

                    record[f"top{i+1}_ly1"] = c["ly1"]
                    record[f"top{i+1}_ly2"] = c["ly2"]
                    record[f"top{i+1}_lx1"] = c["lx1"]
                    record[f"top{i+1}_lx2"] = c["lx2"]

                    record[f"top{i+1}_y1"] = c["y1"]
                    record[f"top{i+1}_y2"] = c["y2"]
                    record[f"top{i+1}_x1"] = c["x1"]
                    record[f"top{i+1}_x2"] = c["x2"]
                else:
                    record[f"top{i+1}_score"] = None
                    record[f"top{i+1}_response_mean"] = None
                    record[f"top{i+1}_monotonic_gain"] = None
                    record[f"top{i+1}_outside_mean"] = None
                    record[f"top{i+1}_lung_ratio"] = None

                    record[f"top{i+1}_ly1"] = None
                    record[f"top{i+1}_ly2"] = None
                    record[f"top{i+1}_lx1"] = None
                    record[f"top{i+1}_lx2"] = None

                    record[f"top{i+1}_y1"] = None
                    record[f"top{i+1}_y2"] = None
                    record[f"top{i+1}_x1"] = None
                    record[f"top{i+1}_x2"] = None

            records.append(record)

            print(
                f"[{label}] {idx+1}/{len(dataset)} "
                f"{case_name} "
                f"top1=img({c1['y1']}:{c1['y2']},{c1['x1']}:{c1['x2']}) "
                f"latent({c1['ly1']}:{c1['ly2']},{c1['lx1']}:{c1['lx2']}) "
                f"valid={valid_blocks}/{total_blocks} "
                f"score={c1['score']:.6f} "
                f"ordinary={ordinary_diff_score:.6f} "
                f"top1={corrected_diff_top1:.6f} "
                f"top3={corrected_diff_top3:.6f} "
                f"top5={corrected_diff_top5:.6f}"
            )

    return records


# ============================================================
# Run
# ============================================================

os.makedirs(save_root, exist_ok=True)

all_records = []

all_records += evaluate_dataset(normal_data_dir, "normal")
all_records += evaluate_dataset(abnormal_data_dir, "abnormal")

df = pd.DataFrame(all_records)

csv_path = os.path.join(
    save_root,
    "stage6_latent_soft_replacement_scores.csv"
)

df.to_csv(csv_path, index=False)

print()
print("Saved CSV:", csv_path)


# ============================================================
# Summary
# ============================================================

score_names = [
    "ordinary_diff",

    "corrected_diff_top1",
    "corrected_diff_top3",
    "corrected_diff_top5",

    "corrected_region_diff_top1",
    "corrected_region_diff_top3",
    "corrected_region_diff_top5",

    "best_response_score",
    "top3_response_score_mean",
    "top5_response_score_mean",

    "response_mean",
    "response_max",

    "qdist_mean",
    "qdist_max",
]

for score_name in score_names:
    print()
    print(f"[{score_name}]")

    normal_scores = df[df["label"] == "normal"][score_name].values
    abnormal_scores = df[df["label"] == "abnormal"][score_name].values

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

    threshold = normal_scores.mean() + 3 * normal_scores.std()

    normal_fp = (normal_scores > threshold).sum()
    abnormal_detected = (abnormal_scores > threshold).sum()

    print(f"Threshold normal mean + 3std: {threshold:.6f}")
    print(f"Normal false positive: {normal_fp} / {len(normal_scores)}")
    print(f"Abnormal detected: {abnormal_detected} / {len(abnormal_scores)}")


print()
print("Each saved image columns:")
print("1 input")
print("2 ordinary recon")
print("3 ordinary diff")
print("4 lung mask")
print("5 latent response map")
print("6 top1 latent-replacement mask")
print("7 top1 corrected recon")
print("8 top1 corrected diff")
print("9 top3 union mask")
print("10 top3 corrected recon")
print("11 top3 corrected diff")
print("12 top5 union mask")
print("13 top5 corrected recon")
print("14 top5 corrected diff")
print("15 qdistance original")

print()
print("CSV saved:")
print(csv_path)