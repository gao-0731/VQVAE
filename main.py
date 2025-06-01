import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import pydicom
from models.vqvae import VQVAE
import utils
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

# =============================
# 超参数设定（全部写死）
# =============================
batch_size = 128
n_updates = 150000
n_hiddens = 128
n_residual_hiddens = 64
n_residual_layers = 2
embedding_dim = 128
n_embeddings = 128
beta = 0.25
learning_rate = 1e-4
log_interval = 50
resize = 192
save = True
filename = "model_checkpoint"
train_data_dir = "/app/data/train"
val_data_dir = "/app/data/validation"

# =============================
# 设备与 TensorBoard 设置
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir=f"./runs/vqvae_{filename}")

# =============================
# DICOM 数据集
# =============================
class DICOMDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform
        self.dicom_files = [os.path.join(root, file)
                            for root, _, files in os.walk(data_path)
                            for file in files if file.endswith('.dcm')]

    def __len__(self):
        return len(self.dicom_files)

    def __getitem__(self, idx):
        dicom_path = self.dicom_files[idx]
        ds = pydicom.dcmread(dicom_path)
        image = ds.pixel_array.astype(np.float32)
        image = np.clip(image, 0, 3072) / 3072.0
        image = torch.tensor(image).unsqueeze(0)
        return self.transform(image) if self.transform else image

# =============================
# 数据加载与变换
# =============================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((resize, resize)),
    transforms.ToTensor()
])

train_loader = DataLoader(DICOMDataset(train_data_dir, transform),
                          batch_size=batch_size, shuffle=True)
val_loader = DataLoader(DICOMDataset(val_data_dir, transform),
                        batch_size=batch_size, shuffle=False)

# =============================
# 模型定义
# =============================
model = VQVAE(
    img_size=192,
    patch_size=4,
    emb_dim=128,
    num_embeddings=128,
    beta=0.25,
    enc_layers=2,
    dec_layers=6,
    use_residual=True,
    n_res_layers=2,
    res_h_dim=64
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

# =============================
# 验证函数
# =============================
def evaluate(step):
    model.eval()
    losses, perplexities = [], []
    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            embedding_loss, x_hat, perplexity = model(x)
            recon_loss = torch.mean((x_hat - x)**2)
            ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=1)
            ms_ssim_error = 1 - ms_ssim_module(x, x_hat)
            losses.append((recon_loss + embedding_loss + ms_ssim_error).item())
            perplexities.append(perplexity.item())

    writer.add_scalar("Val/Loss", np.mean(losses), step)
    writer.add_scalar("Val/Perplexity", np.mean(perplexities), step)

    sample = next(iter(val_loader)).to(device)
    _, x_hat, _ = model(sample)
    grid = vutils.make_grid(torch.cat([sample[:8], x_hat[:8]]), nrow=8, normalize=True)
    writer.add_image("Val/Reconstruction", grid, step)
    return np.mean(losses)

# =============================
# 训练主循环
# =============================
def train():
    best_loss = float("inf")
    for i in range(n_updates):
        model.train()
        x = next(iter(train_loader)).to(device)

        optimizer.zero_grad()
        embedding_loss, x_hat, perplexity = model(x)
        recon_loss = torch.mean((x_hat - x) ** 2)
        ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=1)
        ms_ssim_error = 1 - ms_ssim_module(x, x_hat)
        loss = recon_loss + embedding_loss + ms_ssim_error
        loss.backward()
        optimizer.step()

        # TensorBoard logging
        writer.add_scalar("Train/Loss", loss.item(), i)
        writer.add_scalar("Train/Reconstruction_Loss", recon_loss.item(), i)
        writer.add_scalar("Train/Perplexity", perplexity.item(), i)

        if i % 500 == 0:
            grid = vutils.make_grid(torch.cat([x[:8], x_hat[:8]]), nrow=8, normalize=True)
            writer.add_image("Train/Reconstruction", grid, i)

        # === 10000イテレーションごとに保存・ベスト更新 ===
        if i % 10000 == 0 and i != 0:
            val_loss = evaluate(i)

            # スナップショットとして保存（例: model_checkpoint_iter10000.pth）
            if save:
                utils.save_model_and_results(
                    model, {"n_updates": i}, vars(), f"{filename}_iter{i}"
                )

                # ベストモデルとして保存（例: model_checkpoint_best.pth）
                if val_loss < best_loss:
                    best_loss = val_loss
                    utils.save_model_and_results(
                        model, {"n_updates": i}, vars(), f"{filename}_best"
                    )

if __name__ == "__main__":
    train()
