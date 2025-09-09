import torch
import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder
from models.quantizer import VectorQuantizer

class VQVAE(nn.Module):
    def __init__(self, img_size=256, patch_size=16, emb_dim=[512, 256, 128], num_embeddings=128, beta=0.25,
                 enc_layers=6, use_residual=True, n_res_layers=3, res_h_dim=64):
        super().__init__()
        self.encoder = Encoder(
            in_channels=1, emb_dims=emb_dim, patch_size=patch_size, img_size=img_size,
            num_layers=enc_layers, use_residual=use_residual,
            n_res_layers=n_res_layers, res_h_dim=res_h_dim
        )
        self.quantizer = VectorQuantizer(num_embeddings, emb_dim[-1], beta)

        self.decoder = Decoder(emb_dim[-1], emb_dim[-1], n_res_layers, res_h_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        loss, z_q, perplexity, _, _ = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        return loss, x_recon, perplexity
    