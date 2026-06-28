import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()

        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(
            self.n_e,
            self.e_dim
        )

        self.embedding.weight.data.uniform_(
            -1.0 / self.n_e,
            1.0 / self.n_e
        )

    def forward(self, z):
        """
        z: [B, C, H, W]
        """

        # [B, C, H, W] -> [B, H, W, C]
        z = z.permute(0, 2, 3, 1).contiguous()

        # [B, H, W, C] -> [B*H*W, C]
        z_flattened = z.view(-1, self.e_dim)

        # distance:
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z e
        d = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(
                z_flattened,
                self.embedding.weight.t()
            )
        )

        # =============================
        # 新增：最小 quantization distance map
        # =============================
        min_distances = torch.min(d, dim=1)[0]

        min_distance_map = min_distances.view(
            z.shape[0],
            z.shape[1],
            z.shape[2]
        )  # [B, H, W]

        # nearest code index
        min_encoding_indices = torch.argmin(
            d,
            dim=1
        ).unsqueeze(1)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0],
            self.n_e,
            device=z.device
        )

        min_encodings.scatter_(
            1,
            min_encoding_indices,
            1
        )

        # quantized latent
        z_q = torch.matmul(
            min_encodings,
            self.embedding.weight
        ).view(z.shape)

        # VQ loss
        loss = (
            torch.mean((z_q.detach() - z) ** 2)
            + self.beta * torch.mean((z_q - z.detach()) ** 2)
        )

        # straight-through estimator
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)

        perplexity = torch.exp(
            -torch.sum(
                e_mean * torch.log(e_mean + 1e-10)
            )
        )

        # [B, H, W, C] -> [B, C, H, W]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return (
            loss,
            z_q,
            perplexity,
            min_encodings,
            min_encoding_indices,
            min_distance_map
        )