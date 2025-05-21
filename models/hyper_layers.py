import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
import numpy as np
#import geoopt

class MLPLayers(nn.Module):
    def __init__(
        self, layers, dropout=0.0, activation="relu", bn=False
    ):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))

            if self.use_bn and idx != (len(self.layers)-2):
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))

            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None and idx != (len(self.layers)-2):
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


def activation_layer(activation_name="relu", emb_dim=None):

    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_name.lower() == "relu":
            activation = nn.ReLU()
        elif activation_name.lower() == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name.lower() == "none":
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            "activation function {} is not implemented".format(activation_name)
        )

    return activation



def hyperbolic_centroid_multi(points):
    if isinstance(points, torch.Tensor):  
        points = points.detach().cpu().numpy()

    minsk_points = poinc_to_minsk_multi(points)  # Convert to Minkowski space
    minsk_centroid = np.mean(minsk_points, axis=0)  # Compute mean in Minkowski space

    # Normalize Minkowski centroid to lie on the hyperboloid
    norm_sq = np.sum(minsk_centroid[:-1]**2) - minsk_centroid[-1]**2
    normalizer = np.sqrt(np.abs(norm_sq))
    minsk_centroid = minsk_centroid / normalizer
    # Convert back to Poincaré space
    return minsk_to_poinc_multi(minsk_centroid.reshape((1, minsk_points.shape[1])))[0]


def poinc_to_minsk_multi(points):
    if isinstance(points, torch.Tensor):  # Ensure input is a NumPy array
        points = points.detach().cpu().numpy()

    norm_sq = np.sum(points**2, axis=1, keepdims=True)  # Compute squared norm of each point
    factor = 2 / (1 - norm_sq)  # Compute scaling factor

    minsk_points = np.zeros((points.shape[0], points.shape[1] + 1))  # Create Minkowski space array
    minsk_points[:, :-1] = factor * points  # First d coordinates
    minsk_points[:, -1] = (1 + norm_sq[:, 0]) / (1 - norm_sq[:, 0])  # Last coordinate

    return minsk_points

def minsk_to_poinc_multi(points):
    if isinstance(points, torch.Tensor): 
        points = points.detach().cpu().numpy()
    poinc_points = points[:, :-1] / (1 + points[:, -1][:, None])  # Normalize using last coordinate
    return poinc_points

def poincare_distance(manifold, x, y):
    return manifold.dist(x, y)

def hyp_kmeans(
        X,
        manifold,
        num_clusters,
        num_iters = 100, 
        tol = 1e-4):
    n_samples, dim = X.shape
    device = X.device

    # Randomly initialize centroids in the Poincaré ball
    indices = torch.randperm(n_samples)[:num_clusters]
    centroids = X[indices].clone()

    for i in range(num_iters):
        distances = torch.stack([poincare_distance(manifold, X, c) for c in centroids])
        labels = torch.argmin(distances, dim=0)

        new_centroids = []
        for k in range(num_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                    #mean = manifold.mean(cluster_points)
                    mean = hyperbolic_centroid_multi(cluster_points)
                    mean = torch.tensor(mean).to(device)
            else:
                mean = centroids[k]

            new_centroids.append(mean)

        new_centroids = torch.stack(new_centroids)
        #print(f'devioce of centroids:{centroids.device}, and device of new one:{new_centroids.device}')
        # Check for convergence
        shift = torch.norm(new_centroids - centroids, dim=1).max().item()
        centroids = new_centroids

        if shift < tol:
            break

    return centroids



@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    Q = torch.exp(- distances / epsilon)

    B = Q.shape[0] # number of samples to assign
    K = Q.shape[1] # how many centroids per block (usually set to 256)

    # make the matrix sums to 1
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q
    # print(Q.sum())
    for it in range(sinkhorn_iterations):

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K


    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q