import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import Encoder
from models.decoder import Decoder
from models.quantizer import VectorQuantizer


def make_block_mask(
    batch_size,
    height,
    width,
    mask_ratio=0.25,
    block_size=8,
    device="cuda"
):
    """
    Generate block mask on latent feature map.

    Args:
        batch_size: batch size
        height: latent height, usually 64
        width: latent width, usually 64
        mask_ratio: approximate masked area ratio
        block_size: block size on latent map
        device: torch device

    Returns:
        mask: [B, 1, H, W], bool tensor
              True means masked position
    """

    mask = torch.zeros(
        batch_size,
        1,
        height,
        width,
        device=device,
        dtype=torch.bool
    )

    total_pixels = height * width
    target_mask_pixels = int(total_pixels * mask_ratio)
    block_pixels = block_size * block_size

    num_blocks = max(
        1,
        target_mask_pixels // block_pixels
    )

    for b in range(batch_size):
        for _ in range(num_blocks):

            top = torch.randint(
                low=0,
                high=max(1, height - block_size + 1),
                size=(1,),
                device=device
            ).item()

            left = torch.randint(
                low=0,
                high=max(1, width - block_size + 1),
                size=(1,),
                device=device
            ).item()

            mask[
                b,
                :,
                top:top + block_size,
                left:left + block_size
            ] = True

    return mask


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

        self.h_dim = h_dim
        self.res_h_dim = res_h_dim
        self.n_res_layers = n_res_layers
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        # =============================
        # Encoder
        # Input:  [B, 1, 256, 256]
        # Output: [B, h_dim, 64, 64]
        # =============================
        self.encoder = Encoder(
            in_dim=1,
            h_dim=h_dim,
            n_res_layers=n_res_layers,
            res_h_dim=res_h_dim
        )

        # =============================
        # 1x1 projection before VQ
        # [B, h_dim, 64, 64] -> [B, embedding_dim, 64, 64]
        # =============================
        self.pre_quantization_conv = nn.Conv2d(
            h_dim,
            embedding_dim,
            kernel_size=1,
            stride=1
        )

        # =============================
        # Position-aware latent bias
        # Current latent size is fixed as 64x64
        # =============================
        self.pos_bias = nn.Parameter(
            torch.zeros(
                1,
                embedding_dim,
                64,
                64
            )
        )

        nn.init.trunc_normal_(
            self.pos_bias,
            std=0.02
        )

        # =============================
        # Vector Quantizer
        # =============================
        self.vector_quantization = VectorQuantizer(
            n_embeddings,
            embedding_dim,
            beta
        )

        # =============================
        # Decoder
        # Input:  [B, embedding_dim, 64, 64]
        # Output: [B, 1, 256, 256]
        # =============================
        self.decoder = Decoder(
            embedding_dim,
            h_dim,
            n_res_layers,
            res_h_dim
        )

    def forward(
        self,
        x,
        mask=False,
        mask_ratio=0.15,
        block_size=8,
        return_mask=False
    ):
        """
        Args:
            x: [B, 1, 256, 256]
            mask:
                False: normal VQ-VAE forward
                True: apply block latent masking
            mask_ratio:
                latent mask ratio
            block_size:
                block size on 64x64 latent feature map
            return_mask:
                True: return image_mask for masked-only loss

        Returns:
            if return_mask=False:
                embedding_loss, x_hat, perplexity, q_distance_map

            if return_mask=True:
                embedding_loss, x_hat, perplexity, q_distance_map, image_mask
        """

        # =============================
        # Encoder
        # =============================
        z_e = self.encoder(x)

        # =============================
        # Projection to embedding dim
        # =============================
        z_e = self.pre_quantization_conv(z_e)

        # =============================
        # Add position bias
        # =============================
        if (
            self.pos_bias.shape[2] == z_e.shape[2]
            and self.pos_bias.shape[3] == z_e.shape[3]
        ):
            z_e = z_e + self.pos_bias
        else:
            # Safety for unexpected latent size
            pos_bias = F.interpolate(
                self.pos_bias,
                size=z_e.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            z_e = z_e + pos_bias

        latent_mask = None
        image_mask = None

        # =============================
        # Block latent masking
        # training: call model(..., mask=True)
        # eval:     call model(..., mask=True) if you want masked inference
        # =============================
        if mask:

            latent_mask = make_block_mask(
                batch_size=z_e.shape[0],
                height=z_e.shape[2],
                width=z_e.shape[3],
                mask_ratio=mask_ratio,
                block_size=block_size,
                device=z_e.device
            )

            # Mask latent feature
            z_e = z_e.masked_fill(
                latent_mask,
                0.0
            )

            # Convert latent mask to image-level mask
            # [B, 1, 64, 64] -> [B, 1, 256, 256]
            image_mask = F.interpolate(
                latent_mask.float(),
                size=x.shape[-2:],
                mode="nearest"
            )

        # =============================
        # Vector Quantization
        # quantizer should return 6 values
        # =============================
        (
            embedding_loss,
            z_q,
            perplexity,
            _,
            _,
            q_distance_map
        ) = self.vector_quantization(z_e)

        # =============================
        # Decoder
        # =============================
        x_hat = self.decoder(z_q)

        if return_mask:
            return (
                embedding_loss,
                x_hat,
                perplexity,
                q_distance_map,
                image_mask
            )

        return (
            embedding_loss,
            x_hat,
            perplexity,
            q_distance_map
        )