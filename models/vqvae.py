
import torch
import torch.nn as nn
import numpy as np
from models.encoder import Encoder
from models.quantizer import VectorQuantizer
from models.decoder import Decoder


class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False):
        super(VQVAE, self).__init__()

        self.encoder = Encoder(1, h_dim, n_res_layers, res_h_dim)

        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)

        # ✅ 新增：latent 的 2D position bias
        self.pos_bias = nn.Parameter(
            torch.zeros(1, embedding_dim, 64, 64)
        )
        nn.init.trunc_normal_(self.pos_bias, std=0.02)

        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)

        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)


    def forward(self, x, verbose=False):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)

        # ✅ 关键：加位置偏置
        z_e = z_e + self.pos_bias

        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        return embedding_loss, x_hat, perplexity
