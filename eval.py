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
batch_size = 16

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
# Mask Parameters
# 必须和你最新训练一致
# =============================

mask_ratio = 0.05
block_size = 4


# =============================
# Paths
# =============================

eval_data_dir = "/app/data/temp/abnormal"

# 改成你最新训练出来的模型
# 先用 final，如果效果不好再试 best
model_path = "/app/results/vqvae_data_vqvae_ct_masked_block_stable_final-1.0.pth"

# 也可以试：
# model_path = "/app/results/vqvae_data_vqvae_ct_masked_block_stable_best.pth"
# model_path = "/app/results/vqvae_data_vqvae_ct_masked_block_stable_epoch50.pth"

checkpoint_name = os.path.basename(model_path).replace(".pth", "")
eval_time = datetime.now().strftime("%Y%m%d_%H%M%S")

save_dir = f"/app/results/eval_masked_{checkpoint_name}_{eval_time}"
os.makedirs(save_dir, exist_ok=True)

writer = SummaryWriter(
    log_dir=f"/app/runs/eval_masked_{checkpoint_name}_{eval_time}"
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
        # 必须和训练时一致
        # 你现在训练用的是这个预处理
        # =============================
        image = np.clip(image, 0, 3072)
        image = image / 3072.0

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

val_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print("Loader batches:", len(val_loader))


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
# Helper: normalize per image
# =============================

def normalize_per_image(x):
    """
    x: [B, 1, H, W]
    return: [B, 1, H, W]
    每张图单独 normalize 到 [0, 1]
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


# =============================
# Evaluation
# =============================

losses = []
masked_losses = []
full_losses = []
embedding_losses = []
perplexities = []

with torch.no_grad():

    for step, batch in enumerate(val_loader):

        x, paths = batch
        x = x.to(device)

        # =============================
        # Forward with masked inference
        # 注意：这里必须 mask=True
        # =============================

        (
            embedding_loss,
            x_hat,
            perplexity,
            q_distance_map,
            image_mask
        ) = model(
            x,
            mask=True,
            mask_ratio=mask_ratio,
            block_size=block_size,
            return_mask=True
        )

        # =============================
        # Loss / Diff
        # =============================

        full_diff = torch.abs(x - x_hat)
        masked_diff = full_diff * image_mask

        masked_loss = masked_diff.sum() / (
            image_mask.sum() + 1e-8
        )

        full_loss = torch.mean(full_diff)

        recon_loss = 0.3 * masked_loss + 1.0 * full_loss

        total_loss = recon_loss + embedding_loss

        losses.append(total_loss.item())
        masked_losses.append(masked_loss.item())
        full_losses.append(full_loss.item())
        embedding_losses.append(embedding_loss.item())
        perplexities.append(perplexity.item())

        # =============================
        # Visualization
        # =============================

        full_diff_vis = normalize_per_image(full_diff)
        masked_diff_vis = normalize_per_image(masked_diff)

        # =============================
        # Quantization distance map
        # q_distance_map: [B, 64, 64]
        # q_map: [B, 1, 256, 256]
        # =============================

        q_map = q_distance_map.unsqueeze(1)

        q_map = F.interpolate(
            q_map,
            size=(resize, resize),
            mode="bilinear",
            align_corners=False
        )

        q_vis = normalize_per_image(q_map)

        # =============================
        # Fusion score
        # masked_diff + qdistance
        # =============================

        score = 0.5 * masked_diff_vis + 0.5 * q_vis
        score_vis = normalize_per_image(score)

        # =============================
        # Save individual images
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
                x_hat[i],
                f"{save_dir}/{base_name}_recon.png",
                normalize=False
            )

            save_image(
                image_mask[i],
                f"{save_dir}/{base_name}_mask.png",
                normalize=False
            )

            save_image(
                masked_diff_vis[i],
                f"{save_dir}/{base_name}_masked_diff.png",
                normalize=False
            )

            save_image(
                full_diff_vis[i],
                f"{save_dir}/{base_name}_full_diff.png",
                normalize=False
            )

            save_image(
                q_vis[i],
                f"{save_dir}/{base_name}_qdistance.png",
                normalize=False
            )

            save_image(
                score_vis[i],
                f"{save_dir}/{base_name}_score.png",
                normalize=False
            )

            with open(
                f"{save_dir}/{base_name}_path.txt",
                "w"
            ) as f:
                f.write(paths[i])

        # =============================
        # TensorBoard image grid
        #
        # 行顺序：
        # input
        # reconstruction
        # mask
        # masked_diff
        # full_diff
        # qdistance
        # score
        # =============================

        n_show = min(8, x.size(0))

        grid = vutils.make_grid(
            torch.cat([
                x[:n_show],
                x_hat[:n_show],
                image_mask[:n_show],
                masked_diff_vis[:n_show],
                full_diff_vis[:n_show],
                q_vis[:n_show],
                score_vis[:n_show]
            ], dim=0),
            nrow=n_show,
            normalize=True
        )

        writer.add_image(
            "Eval/Input_Recon_Mask_MaskedDiff_FullDiff_QDistance_Score",
            grid,
            step
        )

        # =============================
        # TensorBoard Scalars
        # =============================

        writer.add_scalar(
            "Eval/Loss",
            total_loss.item(),
            step
        )

        writer.add_scalar(
            "Eval/Reconstruction_Loss",
            recon_loss.item(),
            step
        )

        writer.add_scalar(
            "Eval/Masked_Loss",
            masked_loss.item(),
            step
        )

        writer.add_scalar(
            "Eval/Full_Loss",
            full_loss.item(),
            step
        )

        writer.add_scalar(
            "Eval/Embedding_Loss",
            embedding_loss.item(),
            step
        )

        writer.add_scalar(
            "Eval/Perplexity",
            perplexity.item(),
            step
        )

        print(
            f"Step {step:04d} | "
            f"Loss {total_loss.item():.6f} | "
            f"Recon {recon_loss.item():.6f} | "
            f"Masked {masked_loss.item():.6f} | "
            f"Full {full_loss.item():.6f} | "
            f"Embed {embedding_loss.item():.6f} | "
            f"Perplexity {perplexity.item():.4f}"
        )


# =============================
# Finish
# =============================

print("==== Evaluation Finished ====")
print(f"Average Loss: {np.mean(losses):.6f}")
print(f"Average Masked Loss: {np.mean(masked_losses):.6f}")
print(f"Average Full Loss: {np.mean(full_losses):.6f}")
print(f"Average Embedding Loss: {np.mean(embedding_losses):.6f}")
print(f"Average Perplexity: {np.mean(perplexities):.6f}")

print("Saved to:", save_dir)
print("Save dir absolute:", os.path.abspath(save_dir))
print("TensorBoard logdir:", f"/app/runs/eval_masked_{checkpoint_name}_{eval_time}")

writer.close()