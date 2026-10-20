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

# 使用 Stage1 full VQ-VAE，避免 Stage2 masked reconstruction 黑点
model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage1_full_final.pth"

save_root = "/app/results/eval_stage5_v4_sliding_topk_candidates"

resize = 256
batch_size = 1
num_workers = 2

# sliding mask 参数
block_size = 40
stride = 16

# 肺内填充值
fill_value = 0.08

# block 至少多少比例在 lung mask 内才参与 sliding
min_lung_ratio = 0.25

# block score 计算方式
# "mean" = block 内平均 diff
# "topk" = block 内 top-k% diff 平均
score_mode = "topk"
topk_ratio = 0.10

# 保存 top-N candidate masks
top_n = 5

# 为了避免太慢，可限制保存图片数量；None 表示全部保存
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
# Utility functions
# ============================================================

def normalize_map(x):
    x = x.detach()
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def extract_recon_and_qdist(model, x):
    """
    x: [B,1,256,256]

    return:
        x_hat: [B,1,256,256]
        q_distance_map: [B,64,64]
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

    x_hat = model.decoder(z_q)

    return x_hat, q_distance_map, perplexity


def make_lung_mask(x):
    """
    x: [1,1,256,256]

    return:
        lung_mask: [1,1,256,256]

    宽松 lung mask，用于限制 sliding 主要在肺区域附近。
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


def apply_block_mask(x, y1, y2, x1, x2):
    """
    对指定 block 使用 lung-gray 填充
    """

    x_masked = x.clone()
    x_masked[:, :, y1:y2, x1:x2] = fill_value

    mask = torch.zeros_like(x)
    mask[:, :, y1:y2, x1:x2] = 1.0

    return x_masked, mask


def compute_block_score(block_diff):
    """
    block_diff: [1,1,block_size,block_size]
    """

    if score_mode == "mean":
        return block_diff.mean().item()

    elif score_mode == "topk":
        flat = block_diff.reshape(-1)
        k = max(1, int(flat.numel() * topk_ratio))
        vals, _ = torch.topk(flat, k)
        return vals.mean().item()

    else:
        raise ValueError(f"Unknown score_mode: {score_mode}")


def make_union_mask_from_candidates(x, candidates):
    """
    candidates: list of dict, each has y1,y2,x1,x2
    return:
        masked_input, union_mask
    """

    x_masked = x.clone()
    union_mask = torch.zeros_like(x)

    for c in candidates:
        y1, y2, x1, x2 = c["y1"], c["y2"], c["x1"], c["x2"]
        x_masked[:, :, y1:y2, x1:x2] = fill_value
        union_mask[:, :, y1:y2, x1:x2] = 1.0

    return x_masked, union_mask


def suppress_nearby_candidates(candidates, iou_threshold=0.30):
    """
    简单 NMS，避免 top5 都集中在几乎同一个位置。
    candidates: score 从高到低排序
    """

    selected = []

    for c in candidates:
        keep = True

        for s in selected:
            yy1 = max(c["y1"], s["y1"])
            yy2 = min(c["y2"], s["y2"])
            xx1 = max(c["x1"], s["x1"])
            xx2 = min(c["x2"], s["x2"])

            inter_h = max(0, yy2 - yy1)
            inter_w = max(0, xx2 - xx1)
            inter = inter_h * inter_w

            area_c = (c["y2"] - c["y1"]) * (c["x2"] - c["x1"])
            area_s = (s["y2"] - s["y1"]) * (s["x2"] - s["x1"])

            union = area_c + area_s - inter + 1e-8
            iou = inter / union

            if iou > iou_threshold:
                keep = False
                break

        if keep:
            selected.append(c)

        if len(selected) >= top_n:
            break

    return selected


def sliding_lung_gray_search_top_candidates(model, x):
    """
    对图像做 sliding lung-gray mask，保存 top1/top3/top5 candidates。

    x: [1,1,256,256]

    return:
        selected_candidates
        sliding_score_map
        lung_mask
        valid_blocks
        total_blocks
    """

    lung_mask = make_lung_mask(x)

    sliding_score_map = torch.zeros_like(x)
    sliding_count_map = torch.zeros_like(x)

    candidates = []

    # 用来判断 block 是否在肺区域内
    lung_block_ratio = F.avg_pool2d(
        lung_mask,
        kernel_size=block_size,
        stride=stride
    )

    block_rows = lung_block_ratio.shape[-2]
    block_cols = lung_block_ratio.shape[-1]

    total_blocks = block_rows * block_cols
    valid_blocks = 0

    with torch.no_grad():
        for by in range(block_rows):
            for bx in range(block_cols):

                lung_ratio = lung_block_ratio[0, 0, by, bx].item()

                if lung_ratio < min_lung_ratio:
                    continue

                y1 = by * stride
                x1 = bx * stride
                y2 = y1 + block_size
                x2 = x1 + block_size

                if y2 > 256 or x2 > 256:
                    continue

                valid_blocks += 1

                x_masked, mask = apply_block_mask(
                    x,
                    y1,
                    y2,
                    x1,
                    x2
                )

                recon_masked, qdist_masked, perp_masked = extract_recon_and_qdist(
                    model,
                    x_masked
                )

                diff = torch.abs(x - recon_masked)

                block_diff = diff[:, :, y1:y2, x1:x2]
                block_score = compute_block_score(block_diff)

                sliding_score_map[:, :, y1:y2, x1:x2] += block_score
                sliding_count_map[:, :, y1:y2, x1:x2] += 1.0

                candidates.append({
                    "score": block_score,
                    "y1": y1,
                    "y2": y2,
                    "x1": x1,
                    "x2": x2,
                    "lung_ratio": lung_ratio,
                })

    # 如果 lung mask 太严格导致没有 valid block，则退回全图 sliding
    if valid_blocks == 0:
        print("Warning: no valid lung blocks. Fallback to full-image sliding.")

        for y1 in range(0, 256 - block_size + 1, stride):
            for x1 in range(0, 256 - block_size + 1, stride):
                y2 = y1 + block_size
                x2 = x1 + block_size

                x_masked, mask = apply_block_mask(
                    x,
                    y1,
                    y2,
                    x1,
                    x2
                )

                recon_masked, qdist_masked, perp_masked = extract_recon_and_qdist(
                    model,
                    x_masked
                )

                diff = torch.abs(x - recon_masked)

                block_diff = diff[:, :, y1:y2, x1:x2]
                block_score = compute_block_score(block_diff)

                sliding_score_map[:, :, y1:y2, x1:x2] += block_score
                sliding_count_map[:, :, y1:y2, x1:x2] += 1.0

                candidates.append({
                    "score": block_score,
                    "y1": y1,
                    "y2": y2,
                    "x1": x1,
                    "x2": x2,
                    "lung_ratio": -1.0,
                })

    sliding_score_map = sliding_score_map / (sliding_count_map + 1e-8)

    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)

    selected_candidates = suppress_nearby_candidates(
        candidates,
        iou_threshold=0.30
    )

    return (
        selected_candidates,
        sliding_score_map,
        lung_mask,
        valid_blocks,
        total_blocks
    )


def reconstruct_with_candidates(model, x, candidates, n_candidate):
    """
    用 top-n candidates 的 union mask 做 targeted reconstruction。

    n_candidate:
        1, 3, 5
    """

    selected = candidates[:n_candidate]

    x_masked, union_mask = make_union_mask_from_candidates(
        x,
        selected
    )

    recon, qdist, perp = extract_recon_and_qdist(
        model,
        x_masked
    )

    diff = torch.abs(x - recon)

    region_diff = (
        diff * union_mask
    ).sum().item() / (union_mask.sum().item() + 1e-8)

    return x_masked, union_mask, recon, diff, region_diff


def topk_mean_score(score_map, k_ratio=0.01):
    """
    score_map: [1,1,H,W]
    计算 top-k mean，用于 case-level score
    """

    flat = score_map.view(-1)
    k = max(1, int(flat.numel() * k_ratio))
    vals, _ = torch.topk(flat, k)
    return vals.mean().item()


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

            # --------------------------------------------------------
            # ordinary reconstruction
            # --------------------------------------------------------
            recon_original, q_distance_map, perplexity = extract_recon_and_qdist(
                model,
                x
            )

            diff_original = torch.abs(x - recon_original)

            # --------------------------------------------------------
            # sliding search top candidates
            # --------------------------------------------------------
            (
                candidates,
                sliding_score_map,
                lung_mask,
                valid_blocks,
                total_blocks
            ) = sliding_lung_gray_search_top_candidates(
                model,
                x
            )

            if len(candidates) == 0:
                print(f"Warning: no candidates found for {case_name}")
                continue

            # --------------------------------------------------------
            # top1 / top3 / top5 reconstruction
            # --------------------------------------------------------
            top1_n = min(1, len(candidates))
            top3_n = min(3, len(candidates))
            top5_n = min(5, len(candidates))

            (
                x_masked_top1,
                mask_top1,
                recon_top1,
                diff_top1,
                region_diff_top1
            ) = reconstruct_with_candidates(
                model,
                x,
                candidates,
                top1_n
            )

            (
                x_masked_top3,
                mask_top3,
                recon_top3,
                diff_top3,
                region_diff_top3
            ) = reconstruct_with_candidates(
                model,
                x,
                candidates,
                top3_n
            )

            (
                x_masked_top5,
                mask_top5,
                recon_top5,
                diff_top5,
                region_diff_top5
            ) = reconstruct_with_candidates(
                model,
                x,
                candidates,
                top5_n
            )

            # --------------------------------------------------------
            # qdistance map upsample
            # --------------------------------------------------------
            q_map = q_distance_map.unsqueeze(1)
            q_map = F.interpolate(
                q_map,
                size=(256, 256),
                mode="bilinear",
                align_corners=False
            )

            # --------------------------------------------------------
            # Scores
            # --------------------------------------------------------
            ordinary_diff_score = diff_original.mean().item()

            targeted_diff_top1 = diff_top1.mean().item()
            targeted_diff_top3 = diff_top3.mean().item()
            targeted_diff_top5 = diff_top5.mean().item()

            sliding_mean = sliding_score_map.mean().item()
            sliding_max = sliding_score_map.max().item()
            sliding_top1percent = topk_mean_score(sliding_score_map, k_ratio=0.01)

            qdist_mean = q_map.mean().item()
            qdist_max = q_map.max().item()

            c1 = candidates[0]
            c3_score = np.mean([c["score"] for c in candidates[:top3_n]])
            c5_score = np.mean([c["score"] for c in candidates[:top5_n]])

            # --------------------------------------------------------
            # Save image
            # --------------------------------------------------------
            should_save = True
            if max_save_images is not None and idx >= max_save_images:
                should_save = False

            image_save_path = ""

            if should_save:
                grid = vutils.make_grid(
                    torch.cat([
                        x.cpu(),                                      # 1 input
                        recon_original.cpu(),                         # 2 ordinary recon
                        normalize_map(diff_original).cpu(),           # 3 ordinary diff
                        lung_mask.cpu(),                              # 4 lung mask
                        normalize_map(sliding_score_map).cpu(),       # 5 sliding score map

                        mask_top1.cpu(),                              # 6 top1 mask
                        x_masked_top1.cpu(),                          # 7 top1 masked input
                        recon_top1.cpu(),                             # 8 top1 recon
                        normalize_map(diff_top1).cpu(),               # 9 top1 diff

                        mask_top3.cpu(),                              # 10 top3 union mask
                        recon_top3.cpu(),                             # 11 top3 recon
                        normalize_map(diff_top3).cpu(),               # 12 top3 diff

                        mask_top5.cpu(),                              # 13 top5 union mask
                        recon_top5.cpu(),                             # 14 top5 recon
                        normalize_map(diff_top5).cpu(),               # 15 top5 diff

                        normalize_map(q_map).cpu(),                   # 16 qdistance original
                    ], dim=0),
                    nrow=16,
                    normalize=True
                )

                image_save_path = os.path.join(
                    save_dir,
                    f"{idx:04d}_{case_name}.png"
                )

                vutils.save_image(grid, image_save_path)

            # --------------------------------------------------------
            # Record
            # --------------------------------------------------------
            record = {
                "label": label,
                "case_name": case_name,
                "path": path,

                "valid_blocks": valid_blocks,
                "total_blocks": total_blocks,

                "ordinary_diff": ordinary_diff_score,

                "targeted_diff_top1": targeted_diff_top1,
                "targeted_diff_top3": targeted_diff_top3,
                "targeted_diff_top5": targeted_diff_top5,

                "targeted_region_diff_top1": region_diff_top1,
                "targeted_region_diff_top3": region_diff_top3,
                "targeted_region_diff_top5": region_diff_top5,

                "best_sliding_score": c1["score"],
                "top3_candidate_score_mean": c3_score,
                "top5_candidate_score_mean": c5_score,

                "sliding_mean": sliding_mean,
                "sliding_max": sliding_max,
                "sliding_top1percent": sliding_top1percent,

                "qdist_mean": qdist_mean,
                "qdist_max": qdist_max,

                "perplexity_original": float(perplexity.item()),

                "image_path": image_save_path,
            }

            # save candidate coordinates
            for i in range(top_n):
                if i < len(candidates):
                    c = candidates[i]
                    record[f"top{i+1}_score"] = c["score"]
                    record[f"top{i+1}_y1"] = c["y1"]
                    record[f"top{i+1}_y2"] = c["y2"]
                    record[f"top{i+1}_x1"] = c["x1"]
                    record[f"top{i+1}_x2"] = c["x2"]
                    record[f"top{i+1}_lung_ratio"] = c["lung_ratio"]
                else:
                    record[f"top{i+1}_score"] = None
                    record[f"top{i+1}_y1"] = None
                    record[f"top{i+1}_y2"] = None
                    record[f"top{i+1}_x1"] = None
                    record[f"top{i+1}_x2"] = None
                    record[f"top{i+1}_lung_ratio"] = None

            records.append(record)

            print(
                f"[{label}] {idx+1}/{len(dataset)} "
                f"{case_name} "
                f"top1=({c1['y1']}:{c1['y2']},{c1['x1']}:{c1['x2']}) "
                f"valid_blocks={valid_blocks}/{total_blocks} "
                f"score={c1['score']:.6f} "
                f"ordinary={ordinary_diff_score:.6f} "
                f"top1_diff={targeted_diff_top1:.6f} "
                f"top3_diff={targeted_diff_top3:.6f} "
                f"top5_diff={targeted_diff_top5:.6f}"
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
    "stage5_v4_sliding_topk_candidates_scores.csv"
)

df.to_csv(csv_path, index=False)

print()
print("Saved CSV:", csv_path)


# ============================================================
# Summary
# ============================================================

score_names = [
    "ordinary_diff",

    "targeted_diff_top1",
    "targeted_diff_top3",
    "targeted_diff_top5",

    "targeted_region_diff_top1",
    "targeted_region_diff_top3",
    "targeted_region_diff_top5",

    "best_sliding_score",
    "top3_candidate_score_mean",
    "top5_candidate_score_mean",

    "sliding_mean",
    "sliding_max",
    "sliding_top1percent",

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


# ============================================================
# Candidate coordinate reminder
# ============================================================

print()
print("Each image columns:")
print("1 input")
print("2 ordinary recon")
print("3 ordinary diff")
print("4 lung mask")
print("5 sliding score map")
print("6 top1 mask")
print("7 top1 masked input")
print("8 top1 recon")
print("9 top1 diff")
print("10 top3 union mask")
print("11 top3 recon")
print("12 top3 diff")
print("13 top5 union mask")
print("14 top5 recon")
print("15 top5 diff")
print("16 qdistance original")

print()
print("CSV saved candidate coordinates:")
print("top1_y1/top1_y2/top1_x1/top1_x2 ... top5_y1/top5_y2/top5_x1/top5_x2")