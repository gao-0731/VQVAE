import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from pytorch_msssim import MS_SSIM
import pydicom
from models.vqvae import VQVAE
from torch.utils.tensorboard import SummaryWriter
from main import DICOMDataset  # 复用训练里的Dataset

# =============================
# 参数（必须和训练一致）
# =============================
resize = 256
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径
val_data_dir = "/app/data/temp/abnormal"
model_path = "./pth_model/vqvae_data_model_checkpoint190000_cnn_attention_embedding.pth"   # ⚠️改成你的新模型路径
save_dir = "./results/reconstruction_new"
os.makedirs(save_dir, exist_ok=True)

# =============================
# 数据处理（必须一致）
# =============================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((resize, resize)),
    transforms.ToTensor()
])

dataset = DICOMDataset(val_data_dir, transform)

print("==== DEBUG ====")
print("Dataset size:", len(dataset))

val_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False
)

print("Loader batches:", len(val_loader))

# =============================
# 模型（和训练完全一致）
# =============================
n_hiddens = 128
n_residual_hiddens = 64
n_residual_layers = 2
embedding_dim = 128
n_embeddings = 128
beta = 0.25

model = VQVAE(
    n_hiddens,
    n_residual_hiddens,
    n_residual_layers,
    n_embeddings,
    embedding_dim,
    beta
).to(device)

# =============================
# 加载模型
# =============================
checkpoint = torch.load(model_path, map_location=device)

# 兼容不同保存格式
if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

model.eval()

# =============================
# 评估
# =============================
ms_ssim_module = MS_SSIM(data_range=1.0, size_average=True, channel=1)

losses = []
perplexities = []

writer = SummaryWriter(log_dir="./runs/eval_new")

with torch.no_grad():
    for step, x in enumerate(val_loader):
        x = x.to(device)

        embedding_loss, x_hat, perplexity = model(x)

        recon_loss = torch.mean((x_hat - x) ** 2)
        ms_ssim_loss = 1 - ms_ssim_module(x, x_hat)

        total_loss = recon_loss + embedding_loss

        losses.append(total_loss.item())
        perplexities.append(perplexity.item())

        # =============================
        # 保存图片
        # =============================
        for i in range(min(16, x.size(0))):
            ori = x[i].repeat(3, 1, 1)
            rec = x_hat[i].repeat(3, 1, 1)

            save_image(
                ori,
                f"{save_dir}/step{step:03d}_img{i:02d}_input.png",
                normalize=False
            )
            save_image(
                rec,
                f"{save_dir}/step{step:03d}_img{i:02d}_recon.png",
                normalize=False
            )

        # =============================
        # TensorBoard
        # =============================
        writer.add_scalar("Eval/Loss", total_loss.item(), step)
        writer.add_scalar("Eval/Reconstruction_Loss", recon_loss.item(), step)
        writer.add_scalar("Eval/MS_SSIM_Loss", ms_ssim_loss.item(), step)
        writer.add_scalar("Eval/Perplexity", perplexity.item(), step)

# =============================
# 输出结果
# =============================
print(f"Average Loss: {np.mean(losses):.6f}")
print(f"Average Perplexity: {np.mean(perplexities):.6f}")
print(f"Saved to: {save_dir}")

print("Current working dir:", os.getcwd())
print("Save dir absolute:", os.path.abspath(save_dir))