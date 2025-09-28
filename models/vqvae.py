import torch
import torch.nn as nn
from models.encoder import ViTEncoder as Encoder
from models.decoder import Decoder
from models.quantizer import VectorQuantizer

class VQVAE(nn.Module):
    def __init__(self, img_size=256, patch_size=16, emb_dim=16, num_embeddings=128, beta=0.25,
                 enc_layers=6, use_residual=True, n_res_layers=3, res_h_dim=64):
        super().__init__()
        self.encoder = Encoder(
            in_channels=1, emb_dim=emb_dim, patch_size=patch_size, image_size=img_size, num_blocks=enc_layers)
        self.quantizer = VectorQuantizer(num_embeddings, emb_dim, beta)

        self.decoder = Decoder(emb_dim, emb_dim, n_res_layers, res_h_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        loss, z_q, perplexity, _, _ = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        return loss, x_recon, perplexity
    