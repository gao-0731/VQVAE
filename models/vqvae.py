import torch
import torch.nn as nn

from models.encoder import Encoder
from models.quantizer import VectorQuantizer
from models.decoder import Decoder


class VQVAE(nn.Module):

    def __init__(
        self,
        h_dim,
        res_h_dim,
        n_res_layers,
        n_embeddings,
        embedding_dim,
        beta
    ):
        super(VQVAE, self).__init__()

        self.encoder = Encoder(
            1,
            h_dim,
            n_res_layers,
            res_h_dim
        )

        self.pre_quantization_conv = nn.Conv2d(
            h_dim,
            embedding_dim,
            kernel_size=1,
            stride=1
        )

        # position-aware latent bias
        self.pos_bias = nn.Parameter(
            torch.zeros(1, embedding_dim, 64, 64)
        )

        nn.init.trunc_normal_(self.pos_bias, std=0.02)

        self.vector_quantization = VectorQuantizer(
            n_embeddings,
            embedding_dim,
            beta
        )

        self.decoder = Decoder(
            embedding_dim,
            h_dim,
            n_res_layers,
            res_h_dim
        )

    def forward(self, x, mask=False, mask_ratio=0.15):

        # =========================
        # Encoder
        # =========================
        z_e = self.encoder(x)

        # =========================
        # 1x1 projection
        # =========================
        z_e = self.pre_quantization_conv(z_e)

        # =========================
        # Position bias
        # =========================
        z_e = z_e + self.pos_bias

        # =========================
        # Latent masking
        # 训练时默认 mask
        # eval 时如果 mask=True，也 mask
        # =========================
        if self.training or mask:

            mask_tensor = (
                torch.rand(
                    z_e.shape[0],
                    1,
                    z_e.shape[2],
                    z_e.shape[3],
                    device=z_e.device
                ) < mask_ratio
            )

            z_e = z_e.masked_fill(mask_tensor, 0)

        # =========================
        # Vector Quantization
        # =========================
        embedding_loss, z_q, perplexity, _, _, q_distance_map = \
            self.vector_quantization(z_e)

        # =========================
        # Decoder
        # =========================
        x_hat = self.decoder(z_q)

        return embedding_loss, x_hat, perplexity, q_distance_map