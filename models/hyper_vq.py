import torch
import torch.nn as nn
import torch.nn.functional as F
#from .layers import kmeans, sinkhorn_algorithm
from .hyper_layers import hyp_kmeans, sinkhorn_algorithm
from hypmath import mobius


class HypVectorQuantizer(nn.Module):

    def __init__(self, n_e, e_dim, manifold,
                 beta = 0.25, kmeans_init = False, kmeans_iters = 10,
                 sk_epsilon=0.003, sk_iters=100,):
        super().__init__()
        self.manifold = manifold
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            print('Initialization with uniform distribution....')
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
            self.embedding.weight.data.copy_(manifold.expmap0(self.embedding.weight.data))
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    # def init_emb(self, data):
    #     centers = kmeans(
    #         data,
    #         self.n_e,
    #         self.kmeans_iters,
    #     )
    #     self.embedding.weight.data.copy_(centers)
    #     self.initted = True

    def init_emb(self, data):
        print('Initialization with kmeans....')
        centers = hyp_kmeans(
            data,
            self.manifold,
            self.n_e,
            self.kmeans_iters,
        )

        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K, preparation for sinkhorn algorithm
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances


    def forward(self, x, use_sk=True):
        #print(f'VQ-using sk or not during training: {use_sk}')
        # Flatten input
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        #Calculate the L2 Norm between latent and Embedded weights
        # d = torch.sum(latent**2, dim=1, keepdim=True) + \
        #     torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
        #     2 * torch.matmul(latent, self.embedding.weight.t())

        latent_norm_sq = torch.sum(latent**2, dim=1, keepdim=True)  # Shape (N, 1)
        embedding_norm_sq = torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()  # Shape (1, M)
        euclidean_sq_dist = torch.sum((latent.unsqueeze(1) - self.embedding.weight.unsqueeze(0)) **2, dim=2)  # Shape (N, M)
        denominator = (1 - latent_norm_sq) * (1 - embedding_norm_sq)  # Shape (N, M)
        d = torch.acosh(1 + 2 * euclidean_sq_dist / denominator)

        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        # indices = torch.argmin(d, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # preserve gradients
        x_q = x + (x_q - x).detach()
        #residual = mobius.mobius_minus(x_q, x).detach()
        #x_q = mobius.mobius_add(x, residual)

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices


