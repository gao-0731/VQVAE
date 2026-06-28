import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import pydicom
from pytorch_msssim import ssim
from models.vqvae import VQVAE
import utils
from datetime import datetime

# =============================
# Hyper Parameters
# =============================
batch_size = 16
n_epochs = 200
n_hiddens = 128
n_residual_hiddens = 64
n_residual_layers = 1
embedding_dim = 64
n_embeddings = 64
beta = 0.25
learning_rate = 1e-4
resize = 256

filename = "vqvae_ct_anomaly"
train_data_dir = "/app/data/train"
val_data_dir = "/app/data/validation"

# =============================
# Device
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(log_dir=f"./runs/{filename}_{run_name}")

# =============================
# Dataset
# =============================
class DICOMDataset(Dataset):

    def __init__(self, data_path, transform=None):
        self.transform = transform
        self.dicom_files = []

        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith(".dcm"):
                    self.dicom_files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.dicom_files)

    def __getitem__(self, idx):
        dicom_path = self.dicom_files[idx]
        ds = pydicom.dcmread(dicom_path)
        image = ds.pixel_array.astype(np.float32)
        image = np.clip(image, 0, 3072)
        image = image / 3072.0
        image = torch.tensor(image).unsqueeze(0)
        if self.transform:
            image = self.transform(image)
        return image

# =============================
# Transform
# =============================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((resize, resize)),
    transforms.ToTensor()
])

# =============================
# DataLoader
# =============================
train_loader = DataLoader(
    DICOMDataset(train_data_dir, transform),
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

val_loader = DataLoader(
    DICOMDataset(val_data_dir, transform),
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

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

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)

# =============================
# Validation
# =============================
def evaluate(epoch):
    model.eval()
    losses = []
    recon_losses = []
    perplexities = []

    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            embedding_loss, x_hat, perplexity = model(x)
            l1_loss = torch.mean(torch.abs(x_hat - x))

            ssim_loss = 1 - ssim(x_hat, x, data_range=1.0)

            recon_loss = l1_loss + 0.1 * ssim_loss
            loss = recon_loss + embedding_loss
            losses.append(loss.item())
            recon_losses.append(recon_loss.item())
            perplexities.append(perplexity.item())

    writer.add_scalar(
        "Val/Loss",
        np.mean(losses),
        epoch
    )

    writer.add_scalar(
        "Val/Reconstruction_Loss",
        np.mean(recon_losses),
        epoch
    )

    writer.add_scalar(
        "Val/Perplexity",
        np.mean(perplexities),
        epoch
    )

    sample = next(iter(val_loader))
    sample = sample.to(device)

    model.eval()

    with torch.no_grad():
        _, x_hat, _ = model(sample)

    grid = vutils.make_grid(
        torch.cat([sample[:8], x_hat[:8]]),
        nrow=8,
        normalize=True
    )

    writer.add_image(
        "Val/Reconstruction",
        grid,
        epoch
    )
    return np.mean(losses)

# =============================
# Train
# =============================
def train():
    best_loss = 999999
    global_step = 0

    for epoch in range(n_epochs):
        model.train()
        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            embedding_loss, x_hat, perplexity = model(x)

            # =====================
            # Reconstruction Loss
            # =====================
            l1_loss = torch.mean(torch.abs(x_hat - x))
            ssim_loss = 1 - ssim(x_hat, x, data_range=1.0)

            recon_loss = l1_loss + 0.1 * ssim_loss
            loss = recon_loss + embedding_loss
            loss.backward()
            optimizer.step()

            # =====================
            # Logging
            # =====================
            writer.add_scalar(
                "Train/Loss",
                loss.item(),
                global_step
            )

            writer.add_scalar(
                "Train/Reconstruction_Loss",
                recon_loss.item(),
                global_step
            )

            writer.add_scalar(
                "Train/Perplexity",
                perplexity.item(),
                global_step
            )

            # =====================
            # Reconstruction Image
            # =====================
            if global_step % 500 == 0:
                grid = vutils.make_grid(
                    torch.cat([
                        x[:8],
                        x_hat[:8]
                    ]),
                    nrow=8,
                    normalize=True
                )

                writer.add_image(
                    "Train/Reconstruction",
                    grid,
                    global_step
                )

            global_step += 1

        # =====================
        # Validation
        # =====================
        val_loss = evaluate(epoch)

        print(
            f"Epoch {epoch} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            utils.save_model_and_results(
                model,
                {"epoch": epoch},
                vars(),
                f"{filename}_best"
            )
        # save every 10 epochs
        if (epoch + 1) % 10 == 0:
            utils.save_model_and_results(
                model,
                {"epoch": epoch},
                vars(),
                f"{filename}_epoch{epoch+1}"
            )

    utils.save_model_and_results(
        model,
        {"epoch": n_epochs},
        vars(),
        f"{filename}_final"
    )

    print("Training Finished")

if __name__ == "__main__":
    train()