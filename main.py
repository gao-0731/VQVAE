import os
import pydicom
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from models.vqvae import VQVAE
import utils
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  # 实现实时更新

# DICOM画像のカスタムデータセットクラス
class DICOMDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.dicom_files = []

        # DICOMファイルを全てリストに追加
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.dcm'):
                    self.dicom_files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.dicom_files)

    def __getitem__(self, idx):
        dicom_path = self.dicom_files[idx]
        try:
            ds = pydicom.dcmread(dicom_path)  # DICOMファイルの読み込み
            image = ds.pixel_array.astype(np.float32)  # ピクセルデータの取得

            # ピクセル値の処理（例：一定の閾値でクリッピング）
            image[image >= 3072] = 3072
            image[image <= -2000] = 0
            image = image / 3072.0  # 正規化

            return image
        except Exception as e:
            print(f"Error reading DICOM file {dicom_path}: {e}")
            return None

parser = argparse.ArgumentParser()

"""
Hyperparameters
"""
# timestamp = utils.readable_timestamp()

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_updates", type=int, default=5000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=64)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64128)
parser.add_argument("--n_embeddings", type=int, default=128)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--log_interval", type=int, default=50)

# データセットのパスを別々に指定
parser.add_argument("--train_data_dir", type=str, default='/path/to/your/train_data')  # 学習データのパス
parser.add_argument("--val_data_dir", type=str, default='/path/to/your/val_data')  # 検証データのパス
parser.add_argument("--resize", type=int, default=256)  # 画像のリサイズサイズ

# whether or not to save model
parser.add_argument("-save", action="store_true")
parser.add_argument("--filename",  type=str, default="model_checkpoint")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.save:
    print(f'Results will be saved in ./results/vqvae_{args.filename}.pth')

"""
Load data and define batch data loaders
"""
# DICOMデータセットの読み込み（学習用）
train_dataset = DICOMDataset(data_path=args.train_data_dir, transform=None)
# DICOMデータセットの読み込み（検証用）
val_dataset = DICOMDataset(data_path=args.val_data_dir, transform=None)

# DataLoaderの設定
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

"""
Set up VQ-VAE model with components defined in ./models/ folder
"""

model = VQVAE(args.n_hiddens, args.n_residual_hiddens,
              args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).to(device)

"""
Set up optimizer and training loop
"""
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

model.train()

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
}

# 検証の関数
def evaluate():
    model.eval()  # モデルを評価モードに変更
    val_recon_errors = []
    val_loss_vals = []
    val_perplexities = []

    with torch.no_grad():  # 勾配計算を無効にする
        for x, _ in val_loader:
            x = x.to(device)

            # VQ-VAEモデルの出力
            embedding_loss, x_hat, perplexity = model(x)
            recon_loss = torch.mean((x_hat - x)**2)
            loss = recon_loss + embedding_loss

            # 結果を保存
            val_recon_errors.append(recon_loss.cpu().detach().numpy())
            val_loss_vals.append(loss.cpu().detach().numpy())
            val_perplexities.append(perplexity.cpu().detach().numpy())

    # 検証結果の平均
    avg_val_recon_error = np.mean(val_recon_errors)
    avg_val_loss = np.mean(val_loss_vals)
    avg_val_perplexity = np.mean(val_perplexities)

    print(f'Validation Results - Recon Error: {avg_val_recon_error}, Loss: {avg_val_loss}, Perplexity: {avg_val_perplexity}')

    return avg_val_loss  # 検証用の損失を返す


# 准备绘图
fig, ax = plt.subplots(3, 1, figsize=(10, 15))  # 3个子图
ax[0].set_title('Reconstruction Error')
ax[1].set_title('Loss')
ax[2].set_title('Perplexity')
ax[0].set_xlabel('Updates')
ax[0].set_ylabel('Reconstruction Error')
ax[1].set_xlabel('Updates')
ax[1].set_ylabel('Loss')
ax[2].set_xlabel('Updates')
ax[2].set_ylabel('Perplexity')

# 初始化空数据
recon_error_line, = ax[0].plot([], [], label='Recon Error')
loss_line, = ax[1].plot([], [], label='Loss')
perplexity_line, = ax[2].plot([], [], label='Perplexity')

# 设定更新范围
x_data, recon_errors, loss_vals, perplexities = [], [], [], []

def update_plot(i):
    # 定期添加新的数据点到x轴和曲线
    x_data.append(i)
    recon_errors.append(np.mean(results["recon_errors"][-args.log_interval:]))
    loss_vals.append(np.mean(results["loss_vals"][-args.log_interval:]))
    perplexities.append(np.mean(results["perplexities"][-args.log_interval:]))

    # 更新每条曲线的数据
    recon_error_line.set_data(x_data, recon_errors)
    loss_line.set_data(x_data, loss_vals)
    perplexity_line.set_data(x_data, perplexities)

    return recon_error_line, loss_line, perplexity_line


def train():
    best_val_loss = float('inf')

    # 初始化动画
    ani = FuncAnimation(fig, update_plot, frames=args.n_updates, interval=100, blit=True)

    for i in range(args.n_updates):
        # 学习数据1个批次
        (x, _) = next(iter(train_loader))
        x = x.to(device)
        optimizer.zero_grad()

        # VQ-VAE模型输出
        embedding_loss, x_hat, perplexity = model(x)
        recon_loss = torch.mean((x_hat - x)**2)
        loss = recon_loss + embedding_loss

        # 反向传播并更新模型
        loss.backward()
        optimizer.step()

        # 记录结果
        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i

        # 输出日志
        if i % args.log_interval == 0:
            if args.save:
                utils.save_model_and_results(model, results, args.__dict__, args.filename)

            print(f"[Update #{i}] Recon Error: {np.mean(results['recon_errors'][-args.log_interval:]):.4f}, "
                  f"Loss: {np.mean(results['loss_vals'][-args.log_interval:]):.4f}, "
                  f"Perplexity: {np.mean(results['perplexities'][-args.log_interval:]):.4f}")

        # 定期进行验证
        if i % 100 == 0:
            val_loss = evaluate()

            # 如果验证损失改善，保存模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if args.save:
                    utils.save_model_and_results(model, results, args.__dict__, args.filename)

    plt.show()  # 显示图形

if __name__ == "__main__":
    train()
