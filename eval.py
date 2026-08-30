import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

import pydicom

from models.vqvae import VQVAE


# =============================
# Eval Parameters
# =============================

resize = 256
batch_size = 1

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =============================
# Model Parameters
# 必须和训练一致
# =============================

n_hiddens = 128
n_residual_hiddens = 64
n_residual_layers = 1

embedding_dim = 64
n_embeddings = 64
beta = 0.25


# =============================
# Sliding Mask Parameters
# =============================

latent_size = 64

block_size = 4

# stride=4 比较快；如果想更细可以改成 2，但会慢很多
stride = 2

scale_factor = resize // latent_size


# =============================
# Post-processing Parameters
# =============================

# sliding score 平滑，减少 block/grid artifact
smooth_kernel_size = 13

# lung mask 阈值
# HU lung window 后 x 在 [0,1]
# 肺野一般偏暗，胸壁/骨头偏亮，外部背景接近 0
lung_lower = 0.03
lung_upper = 0.55

lung_mask_smooth_kernel = 7
lung_mask_threshold = 0.3


# =============================
# Paths
# =============================

eval_data_dir = "/app/data/temp/abnormal"

model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage2_lightmask_final.pth"

# 如果 final 效果不好，也可以试：
# model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage2_lightmask_best.pth"
# model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage2_lightmask_epoch50.pth"
# model_path = "/app/results/vqvae_data_vqvae_ct_lungwindow_stage1_full_final.pth"

checkpoint_name = os.path.basename(model_path).replace(".pth", "")
eval_time = datetime.now().strftime("%Y%m%d_%H%M%S")

save_dir = f"/app/results/eval_sliding_processed_{checkpoint_name}_{eval_time}"
os.makedirs(save_dir, exist_ok=True)

writer = SummaryWriter(
    log_dir=f"/app/runs/eval_sliding_processed_{checkpoint_name}_{eval_time}"
)

print("Eval data dir:", eval_data_dir)
print("Model path:", model_path)
print("Save dir:", save_dir)


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
                    self.dicom_files.append(
                        os.path.join(root, file)
                    )

        self.dicom_files = sorted(self.dicom_files)

        print(f"Loaded {len(self.dicom_files)} DICOM files from {data_path}")

    def __len__(self):
        return len(self.dicom_files)

    def __getitem__(self, idx):

        dicom_path = self.dicom_files[idx]

        ds = pydicom.dcmread(dicom_path)

        image = ds.pixel_array.astype(np.float32)

        # =============================
        # HU conversion
        # 必须和 Stage1/Stage2 训练一致
        # =============================

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))

        image = image * slope + intercept

        # =============================
        # Lung window
        # 必须和训练一致
        # =============================

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
# 必须和训练一致
# =============================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((resize, resize)),
    transforms.ToTensor()
])


# =============================
# DataLoader
# =============================

dataset = DICOMDataset(
    eval_data_dir,
    transform=transform
)

print("Dataset size:", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print("Loader batches:", len(loader))


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


# =============================
# Load Checkpoint
# =============================

checkpoint = torch.load(
    model_path,
    map_location=device
)

if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

model.eval()

print("Loaded model:", model_path)


# =============================
# Helpers
# =============================

def normalize_per_image(x):
    """
    x: [B, 1, H, W]
    每张图单独 normalize 到 [0,1]
    """

    x_norm = x.clone()

    for i in range(x_norm.size(0)):
        img = x_norm[i]
        img_min = img.min()
        img_max = img.max()

        x_norm[i] = (img - img_min) / (
            img_max - img_min + 1e-8
        )

    return x_norm


def make_single_block_mask(
    batch_size,
    latent_h,
    latent_w,
    top,
    left,
    block_size,
    device
):
    """
    制作一个固定位置 latent block mask

    return:
        latent_mask: [B, 1, latent_h, latent_w]
    """

    latent_mask = torch.zeros(
        batch_size,
        1,
        latent_h,
        latent_w,
        device=device,
        dtype=torch.float32
    )

    latent_mask[
        :,
        :,
        top:top + block_size,
        left:left + block_size
    ] = 1.0

    return latent_mask


def forward_with_given_latent_mask(
    model,
    x,
    latent_mask
):
    """
    手动给定 latent_mask，做 sliding block inference。

    x: [B,1,256,256]
    latent_mask: [B,1,64,64]
    """

    # =============================
    # Encoder
    # =============================

    z_e = model.encoder(x)
    z_e = model.pre_quantization_conv(z_e)

    # =============================
    # Position bias
    # =============================

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

    # =============================
    # Apply given latent mask
    # =============================

    latent_mask_bool = latent_mask.bool()

    z_e_masked = z_e.masked_fill(
        latent_mask_bool,
        0.0
    )

    # =============================
    # Quantization
    # quantizer 需要已经返回 6 个值
    # =============================

    (
        embedding_loss,
        z_q,
        perplexity,
        _,
        _,
        q_distance_map
    ) = model.vector_quantization(z_e_masked)

    # =============================
    # Decoder
    # =============================

    x_hat = model.decoder(z_q)

    image_mask = F.interpolate(
        latent_mask,
        size=x.shape[-2:],
        mode="nearest"
    )

    return (
        embedding_loss,
        x_hat,
        perplexity,
        q_distance_map,
        image_mask
    )


def make_lung_mask(x):
    """
    x: HU lung window 后的图像, [B,1,256,256], range [0,1]

    思路：
    1. 肺内空气区域在 lung window 下通常比较暗
    2. 先取暗区域作为 lung core
    3. 用 max_pool 扩张 lung core，保留肺边缘和贴近肺的病灶
    """

    # 肺野核心区域：偏暗区域
    lung_core = (
        (x > 0.03) &
        (x < 0.45)
    ).float()

    # 平滑一下，去掉零散噪声
    lung_core = F.avg_pool2d(
        lung_core,
        kernel_size=5,
        stride=1,
        padding=2
    )

    lung_core = (lung_core > 0.3).float()

    # 扩张肺区域，保留肺边缘、胸膜附近病灶
    lung_mask = F.max_pool2d(
        lung_core,
        kernel_size=21,
        stride=1,
        padding=10
    )

    return lung_mask


def smooth_score(score):
    """
    平滑 anomaly score，减少 block/grid artifact。
    score: [B,1,H,W]
    """

    score = F.avg_pool2d(
        score,
        kernel_size=smooth_kernel_size,
        stride=1,
        padding=smooth_kernel_size // 2
    )

    return score


# =============================
# Evaluation
# =============================

with torch.no_grad():

    for step, batch in enumerate(loader):

        x, paths = batch
        x = x.to(device)

        b = x.size(0)

        # =============================
        # 普通完整重建：参考用
        # =============================

        (
            embedding_loss_full,
            x_hat_full,
            perplexity_full,
            q_distance_map_full
        ) = model(
            x,
            mask=False
        )

        full_diff = torch.abs(x - x_hat_full)
        full_diff_vis = normalize_per_image(full_diff)

        # =============================
        # Full qdistance
        # =============================

        q_map = q_distance_map_full.unsqueeze(1)

        q_map = F.interpolate(
            q_map,
            size=(resize, resize),
            mode="bilinear",
            align_corners=False
        )

        # =============================
        # Lung mask
        # =============================

        lung_mask = make_lung_mask(x)

        # =============================
        # Sliding block anomaly map
        # =============================

        sliding_score = torch.zeros_like(x)
        count_map = torch.zeros_like(x)

        recon_sum = torch.zeros_like(x)
        recon_count = torch.zeros_like(x)

        num_blocks = 0

        for top in range(0, latent_size - block_size + 1, stride):
            for left in range(0, latent_size - block_size + 1, stride):

                latent_mask = make_single_block_mask(
                    batch_size=b,
                    latent_h=latent_size,
                    latent_w=latent_size,
                    top=top,
                    left=left,
                    block_size=block_size,
                    device=device
                )

                (
                    embedding_loss,
                    x_hat,
                    perplexity,
                    q_distance_map,
                    image_mask
                ) = forward_with_given_latent_mask(
                    model,
                    x,
                    latent_mask
                )

                diff = torch.abs(x - x_hat)

                # 只把当前 mask 区域的 diff 写入 score
                sliding_score += diff * image_mask
                count_map += image_mask

                recon_sum += x_hat * image_mask
                recon_count += image_mask

                num_blocks += 1

        sliding_score = sliding_score / (
            count_map + 1e-8
        )

        sliding_recon = recon_sum / (
            recon_count + 1e-8
        )

        # 没覆盖到的地方，用 full reconstruction 补上
        uncovered = (recon_count == 0).float()
        sliding_recon = sliding_recon * (1.0 - uncovered) + x_hat_full * uncovered

        # =============================
        # Raw visualization before post-process
        # =============================

        sliding_score_raw_vis = normalize_per_image(sliding_score)
        q_raw_vis = normalize_per_image(q_map)

        # =============================
        # Post-process score
        # 1. smooth
        # 2. lung mask
        # =============================

        sliding_score_processed = smooth_score(sliding_score)

        q_processed = q_map

        full_diff_processed = smooth_score(full_diff)

        # =============================
        # Normalize processed maps
        # =============================

        sliding_score_processed_vis = normalize_per_image(
            sliding_score_processed
        )

        q_processed_vis = normalize_per_image(
            q_processed
        )

        full_diff_processed_vis = normalize_per_image(
            full_diff_processed
        )

        lung_mask_vis = lung_mask

        # =============================
        # Fusion score
        # =============================

        fusion_score_processed = (
            0.7 * sliding_score_processed_vis
            + 0.3 * q_processed_vis
        )

        fusion_score_processed_vis = normalize_per_image(
            fusion_score_processed
        )

        # =============================
        # Save images
        # =============================

        for i in range(x.size(0)):

            global_idx = step * batch_size + i
            base_name = f"img{global_idx:04d}"

            save_image(
                x[i],
                f"{save_dir}/{base_name}_input.png",
                normalize=False
            )

            save_image(
                x_hat_full[i],
                f"{save_dir}/{base_name}_full_recon.png",
                normalize=False
            )

            save_image(
                sliding_recon[i],
                f"{save_dir}/{base_name}_sliding_recon.png",
                normalize=False
            )

            save_image(
                full_diff_vis[i],
                f"{save_dir}/{base_name}_full_diff_raw.png",
                normalize=False
            )

            save_image(
                full_diff_processed_vis[i],
                f"{save_dir}/{base_name}_full_diff_processed.png",
                normalize=False
            )

            save_image(
                sliding_score_raw_vis[i],
                f"{save_dir}/{base_name}_sliding_score_raw.png",
                normalize=False
            )

            save_image(
                sliding_score_processed_vis[i],
                f"{save_dir}/{base_name}_sliding_score_processed.png",
                normalize=False
            )

            save_image(
                q_raw_vis[i],
                f"{save_dir}/{base_name}_qdistance_raw.png",
                normalize=False
            )

            save_image(
                q_processed_vis[i],
                f"{save_dir}/{base_name}_qdistance_processed.png",
                normalize=False
            )

            save_image(
                fusion_score_processed_vis[i],
                f"{save_dir}/{base_name}_fusion_score_processed.png",
                normalize=False
            )

            save_image(
                lung_mask_vis[i],
                f"{save_dir}/{base_name}_lung_mask.png",
                normalize=False
            )

            with open(
                f"{save_dir}/{base_name}_path.txt",
                "w"
            ) as f:
                f.write(paths[i])

        # =============================
        # TensorBoard grid
        #
        # 行顺序：
        # input
        # full reconstruction
        # sliding reconstruction
        # lung mask
        # full diff processed
        # sliding score raw
        # sliding score processed
        # qdistance processed
        # fusion score processed
        # =============================

        n_show = min(4, x.size(0))

        grid = vutils.make_grid(
            torch.cat([
                x[:n_show],
                x_hat_full[:n_show],
                sliding_recon[:n_show],
                lung_mask_vis[:n_show],
                full_diff_processed_vis[:n_show],
                sliding_score_raw_vis[:n_show],
                sliding_score_processed_vis[:n_show],
                q_processed_vis[:n_show],
                fusion_score_processed_vis[:n_show]
            ], dim=0),
            nrow=n_show,
            normalize=True
        )

        writer.add_image(
            "Eval/Input_FullRecon_SlidingRecon_LungMask_FullDiff_SlidingRaw_SlidingProcessed_QProcessed_Fusion",
            grid,
            step
        )

        # =============================
        # Scalars
        # =============================

        writer.add_scalar(
            "Eval/Full_Embedding_Loss",
            embedding_loss_full.item(),
            step
        )

        writer.add_scalar(
            "Eval/Full_Perplexity",
            perplexity_full.item(),
            step
        )

        writer.add_scalar(
            "Eval/SlidingScore_Raw_Mean",
            sliding_score.mean().item(),
            step
        )

        writer.add_scalar(
            "Eval/SlidingScore_Raw_Max",
            sliding_score.max().item(),
            step
        )

        writer.add_scalar(
            "Eval/SlidingScore_Processed_Mean",
            sliding_score_processed.mean().item(),
            step
        )

        writer.add_scalar(
            "Eval/SlidingScore_Processed_Max",
            sliding_score_processed.max().item(),
            step
        )

        writer.add_scalar(
            "Eval/QDistance_Processed_Mean",
            q_processed.mean().item(),
            step
        )

        writer.add_scalar(
            "Eval/QDistance_Processed_Max",
            q_processed.max().item(),
            step
        )

        print(
            f"Step {step:04d} | "
            f"Blocks {num_blocks} | "
            f"Raw mean {sliding_score.mean().item():.6f} | "
            f"Raw max {sliding_score.max().item():.6f} | "
            f"Processed mean {sliding_score_processed.mean().item():.6f} | "
            f"Processed max {sliding_score_processed.max().item():.6f} | "
            f"Perplexity {perplexity_full.item():.4f}"
        )


# =============================
# Finish
# =============================

print("==== Sliding Processed Evaluation Finished ====")
print("Saved to:", save_dir)
print("Save dir absolute:", os.path.abspath(save_dir))
print("TensorBoard logdir:", f"/app/runs/eval_sliding_processed_{checkpoint_name}_{eval_time}")

writer.close()