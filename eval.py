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
from pytorch_msssim import ssim

from models.vqvae import VQVAE


# =============================
# Eval Parameters
# =============================

resize = 256
batch_size = 16

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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
# Paths
# =============================

eval_data_dir = "/app/data/temp/abnormal"

# 这里改成你想测试的 checkpoint
model_path = "/app/results/vqvae_data_vqvae_ct_anomaly_epoch200.pth"
# model_path = "/app/results/vqvae_data_vqvae_ct_anomaly_epoch100.pth"
# model_path = "/app/results/vqvae_data_vqvae_ct_anomaly_final.pth"

checkpoint_name = os.path.basename(model_path).replace(".pth", "")

eval_time = datetime.now().strftime("%Y%m%d_%H%M%S")

save_dir = f"/app/results/reconstruction_{checkpoint_name}_{eval_time}"
os.makedirs(save_dir, exist_ok=True)

writer = SummaryWriter(
    log_dir=f"/app/runs/eval_{checkpoint_name}_{eval_time}"
)


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

    def __len__(self):
        return len(self.dicom_files)

    def __getitem__(self, idx):
        dicom_path = self.dicom_files[idx]

        ds = pydicom.dcmread(dicom_path)
        image = ds.pixel_array.astype(np.float32)

        # 必须和训练时一致
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
    transform
)

print("==== DEBUG ====")
print("Eval data dir:", eval_data_dir)
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
    return: [B, 1, H, W], each image normalized to [0, 1]
    """

    x_norm = x.clone()

    for i in range(x_norm.size(0)):
        img = x_norm[i]
        img_min = img.min()
        img_max = img.max()
        x_norm[i] = (img - img_min) / (img_max - img_min + 1e-8)

    return x_norm


# =============================
# Evaluation
# =============================

losses = []
recon_losses = []
perplexities = []

with torch.no_grad():

    for step, batch in enumerate(val_loader):

        x, paths = batch
        x = x.to(device)

        # =============================
        # Forward
        # mask=False:
        # 先不要 test-time masking
        # 重点看 q_distance_map
        # =============================

        embedding_loss, x_hat, perplexity, q_distance_map = model(
            x,
            mask=False
        )

        # =============================
        # Loss
        # =============================

        l1_loss = torch.mean(
            torch.abs(x_hat - x)
        )

        ssim_loss = 1 - ssim(
            x_hat,
            x,
            data_range=1.0
        )

        recon_loss = l1_loss + 0.1 * ssim_loss
        total_loss = recon_loss + embedding_loss

        losses.append(total_loss.item())
        recon_losses.append(recon_loss.item())
        perplexities.append(perplexity.item())

        # =============================
        # Pixel difference map
        # =============================

        diff = torch.abs(x - x_hat)
        diff_vis = normalize_per_image(diff)

        # =============================
        # Quantization distance map
        # q_distance_map: [B, 64, 64]
        # upsample -> [B, 1, 256, 256]
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
        # 可选：pixel diff + q distance
        # =============================

        score_vis = 0.3 * diff_vis + 0.7 * q_vis
        score_vis = normalize_per_image(score_vis)

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
                diff_vis[i],
                f"{save_dir}/{base_name}_diff.png",
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

            # 保存原始路径，方便追踪
            with open(
                f"{save_dir}/{base_name}_path.txt",
                "w"
            ) as f:
                f.write(paths[i])

        # =============================
        # TensorBoard image grid
        # 行顺序：
        # input / reconstruction / diff / qdistance / fusion score
        # =============================

        n_show = min(8, x.size(0))

        grid = vutils.make_grid(
            torch.cat([
                x[:n_show],
                x_hat[:n_show],
                diff_vis[:n_show],
                q_vis[:n_show],
                score_vis[:n_show]
            ]),
            nrow=n_show,
            normalize=True
        )

        writer.add_image(
            "Eval/Input_Recon_Diff_QDistance_Score",
            grid,
            step
        )

        # =============================
        # TensorBoard scalars
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
            "Eval/Perplexity",
            perplexity.item(),
            step
        )

        writer.add_scalar(
            "Eval/Embedding_Loss",
            embedding_loss.item(),
            step
        )

        print(
            f"Step {step:04d} | "
            f"Loss {total_loss.item():.6f} | "
            f"Recon {recon_loss.item():.6f} | "
            f"Perplexity {perplexity.item():.4f}"
        )


# =============================
# Finish
# =============================

print("==== Evaluation Finished ====")
print(f"Average Loss: {np.mean(losses):.6f}")
print(f"Average Reconstruction Loss: {np.mean(recon_losses):.6f}")
print(f"Average Perplexity: {np.mean(perplexities):.6f}")
print("Saved to:", save_dir)
print("Save dir absolute:", os.path.abspath(save_dir))

writer.close()