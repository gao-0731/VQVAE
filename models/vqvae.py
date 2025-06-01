import torch
import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder
from models.quantizer import VectorQuantizer

# class VQVAE(nn.Module):
#     def __init__(self, img_size=256, patch_size=16, emb_dim=128, num_embeddings=128, beta=0.25,
#                  enc_layers=6, dec_layers=6, use_residual=True, n_res_layers=3, res_h_dim=64):
#         super().__init__()
#         self.encoder = Encoder(
#             in_channels=1, emb_dim=emb_dim, patch_size=patch_size, img_size=img_size,
#             num_layers=enc_layers, use_residual=use_residual,
#             n_res_layers=n_res_layers, res_h_dim=res_h_dim
#         )
#         self.pre_quantization_conv = nn.Conv2d(
#             num_embeddings, emb_dim, kernel_size=1, stride=1)
#         self.quantizer = VectorQuantizer(num_embeddings, emb_dim, beta)
#         # self.decoder = Decoder(
#         #     out_channels=1, emb_dim=emb_dim, patch_size=patch_size, img_size=img_size,
#         #     num_layers=dec_layers, use_residual=use_residual,
#         #     n_res_layers=n_res_layers, res_h_dim=res_h_dim
#         # )

#         self.decoder = Decoder(emb_dim, num_embeddings, n_res_layers, res_h_dim)

class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(1, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    # def forward(self, x):
    #     z_e = self.encoder(x)
    #     loss, z_q, perplexity, _, _ = self.quantizer(z_e)
    #     x_recon = self.decoder(z_q)
    #     return loss, x_recon, perplexity
    def forward(self, x, verbose=False):

        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity