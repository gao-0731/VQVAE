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

# =============================
# 超参数设定（全部写死）
# =============================
batch_size = 64
n_updates = 5000
n_hiddens = 128
n_residual_hiddens = 64
n_residual_layers = 2
embedding_dim = 128
n_embeddings = 128
beta = 0.25
learning_rate = 1e-4
log_interval = 50
resize = 256
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
model = VQVAE(n_hiddens, n_residual_hiddens, n_residual_layers,
              n_embeddings, embedding_dim, beta).to(device)
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
            losses.append((recon_loss + embedding_loss).item())
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
        recon_loss = torch.mean((x_hat - x)**2)
        loss = recon_loss + embedding_loss
        loss.backward()
        optimizer.step()

        # TensorBoard logging
        writer.add_scalar("Train/Loss", loss.item(), i)
        writer.add_scalar("Train/Reconstruction_Loss", recon_loss.item(), i)
        writer.add_scalar("Train/Perplexity", perplexity.item(), i)

        if i % 500 == 0:
            grid = vutils.make_grid(torch.cat([x[:8], x_hat[:8]]), nrow=8, normalize=True)
            writer.add_image("Train/Reconstruction", grid, i)

        if i % 100 == 0:
            val_loss = evaluate(i)
            if save and val_loss < best_loss:
                best_loss = val_loss
                utils.save_model_and_results(model, {"n_updates": i}, vars(), filename)

if __name__ == "__main__":
    train()
