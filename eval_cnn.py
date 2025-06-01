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
from main_cnn import DICOMDataset  # DICOMDatasetをmain.pyから利用
import matplotlib.pyplot as plt


# =============================
# 設定
# =============================
resize = 256
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_data_dir = "/app/data/temp"
model_path = "./Result/vqvae_data_model_checkpoint_iter20000.pth"
save_dir = "./results/reconstruction"
os.makedirs(save_dir, exist_ok=True)

# =============================
# データ変換とローダー
# =============================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((resize, resize)),
    transforms.ToTensor()
])

val_loader = DataLoader(
    DICOMDataset(val_data_dir, transform),
    batch_size=batch_size, shuffle=False
)

# =============================
# モデルの構築と読み込み
# =============================
model = VQVAE(h_dim=128, res_h_dim=64, n_res_layers=2,
              n_embeddings=128, embedding_dim=128, beta=0.25).to(device)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

# =============================
# 評価処理
# =============================
def save_grayscale_image(tensor_img, path):
    """
    1chのtorch.Tensor画像をmatplotlibを使ってグレースケールで保存
    """
    np_img = tensor_img.squeeze().cpu().numpy()  # [1,H,W] → [H,W]
    plt.imsave(path, np_img, cmap='gray', vmin=0, vmax=1)

ms_ssim_module = MS_SSIM(data_range=1.0, size_average=True, channel=1)
losses, perplexities = [], []
writer = SummaryWriter(log_dir="./runs/eval")

with torch.no_grad():
    for step, x in enumerate(val_loader):
        x = x.to(device)
        embedding_loss, x_hat, perplexity = model(x)

        recon_loss = torch.mean((x_hat - x) ** 2)
        # ms_ssim_loss = 1 - ms_ssim_module(x, x_hat)
        total_loss = recon_loss + embedding_loss

        losses.append(total_loss.item())
        perplexities.append(perplexity.item())

        # 横並びにして保存（8枚分）
        pairs = []
        # 1枚ずつ保存（最大8枚）
        for i in range(min(16, x.size(0))):
            ori = x[i].repeat(3, 1, 1) if x[i].shape[0] == 1 else x[i]
            rec = x_hat[i].repeat(3, 1, 1) if x_hat[i].shape[0] == 1 else x_hat[i]

            save_image(ori, f"{save_dir}/step{step:03d}_img{i:02d}_input.png", normalize=True)
            save_image(rec, f"{save_dir}/step{step:03d}_img{i:02d}_recon.png", normalize=False)

            # ori = x[i]  # ← repeatしない
            # rec = x_hat[i]
            # # 保存先パス
            # input_path = f"{save_dir}/step{step:03d}_img{i:02d}_input.png"
            # recon_path = f"{save_dir}/step{step:03d}_img{i:02d}_recon.png"

            # # 保存（グレースケールで）
            # save_grayscale_image(ori, input_path)
            # save_grayscale_image(rec, recon_path)


        # TensorBoardログ記録
        writer.add_scalar("Eval/Loss", total_loss.item(), step)
        writer.add_scalar("Eval/Perplexity", perplexity.item(), step)

# =============================
# 結果出力
# =============================
print(f"Average Evaluation Loss: {np.mean(losses):.6f}")
print(f"Average Perplexity: {np.mean(perplexities):.6f}")
print(f"Results saved to {save_dir}")