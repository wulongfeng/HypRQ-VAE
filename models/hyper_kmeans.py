import torch
import geoopt
import numpy as np
from collections import defaultdict

class HyperbolicKMeans:
    def __init__(self, n_clusters=2, max_iter=100, tol=1e-4, manifold=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.manifold = manifold if manifold else geoopt.PoincareBall()
        self.centroids = None

    def poincare_distance(self, x, y):
        """Compute the Poincaré distance between points x and y."""
        return self.manifold.dist(x, y)

    def fit(self, X):
        """
        X: Tensor of shape (n_samples, dim), assumed to be in hyperbolic space.
        """
        X = X.to(torch.float32)
        n_samples, dim = X.shape
        device = X.device

        # Randomly initialize centroids in the Poincaré ball
        indices = torch.randperm(n_samples)[:self.n_clusters]
        self.centroids = X[indices].clone()

        for i in range(self.max_iter):
            # Step 1: Assign points to the closest centroid
            distances = torch.stack([self.poincare_distance(X, c) for c in self.centroids])
            labels = torch.argmin(distances, dim=0)

            # Step 2: Update centroids using the Fréchet mean
            new_centroids = []
            for k in range(self.n_clusters):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    #mean = self.manifold.mean(cluster_points)
                    mean = self._hyperbolic_centroid_multi(cluster_points)
                    mean = torch.tensor(mean)
                else:
                    mean = self.centroids[k]  # If no points assigned, keep old centroid
                new_centroids.append(mean)
                #print(f'new_centroids:{new_centroids}')
    
            new_centroids = torch.stack(new_centroids)

            # Check for convergence
            shift = torch.norm(new_centroids - self.centroids, dim=1).max().item()
            self.centroids = new_centroids

            if shift < self.tol:
                break  # Convergence reached

        return self.centroids, labels

    # def compute_frechet_mean(self, points, lr=0.1, max_iter=100, tol=1e-4):
    #     """
    #     Compute the Fréchet mean (centroid) in the Poincaré ball using gradient descent.
    #     """
    #     mean = points.mean(dim=0)  # Start with Euclidean mean as initialization
    #     for _ in range(max_iter):
    #         grad = self.manifold.logmap(mean, points).mean(dim=0)  # Compute tangent space gradient
    #         mean = self.manifold.expmap(mean, -lr * grad)  # Move mean in hyperbolic space
    #         if grad.norm() < tol:
    #             break
    #     return mean
    
    def hyperbolic_centroid(self, points):
        minsk_points = self._poinc_to_minsk(points)
        minsk_centroid = np.mean(minsk_points,axis=0)
        normalizer = np.sqrt(np.abs(minsk_centroid[0]**2+minsk_centroid[1]**2-minsk_centroid[2]**2))
        minsk_centroid = minsk_centroid/normalizer
        return self._minsk_to_poinc(minsk_centroid.reshape((1,3)))[0]
    
    def _poinc_to_minsk(self,points):
        minsk_points = np.zeros((points.shape[0],3))
        minsk_points[:,0] = np.apply_along_axis(arr=points,axis=1,func1d=lambda v: 2*v[0]/(1-v[0]**2-v[1]**2))
        minsk_points[:,1] = np.apply_along_axis(arr=points,axis=1,func1d=lambda v: 2*v[1]/(1-v[0]**2-v[1]**2))
        minsk_points[:,2] = np.apply_along_axis(arr=points,axis=1,func1d=lambda v: (1+v[0]**2+v[1]**2)/(1-v[0]**2-v[1]**2))
        return minsk_points

    def _minsk_to_poinc(self,points):
        poinc_points = np.zeros((points.shape[0],2))
        poinc_points[:,0] = points[:,0]/(1+points[:,2])
        poinc_points[:,1] = points[:,1]/(1+points[:,2])
        return poinc_points


    def _hyperbolic_centroid_multi(self, points):
        if isinstance(points, torch.Tensor):  # Ensure input is a NumPy array
            points = points.detach().cpu().numpy()

        minsk_points = self._poinc_to_minsk_multi(points)  # Convert to Minkowski space
        minsk_centroid = np.mean(minsk_points, axis=0)  # Compute mean in Minkowski space

        # Normalize Minkowski centroid to lie on the hyperboloid
        norm_sq = np.sum(minsk_centroid[:-1]**2) - minsk_centroid[-1]**2
        normalizer = np.sqrt(np.abs(norm_sq))
        minsk_centroid = minsk_centroid / normalizer
        # Convert back to Poincaré space
        return self._minsk_to_poinc_multi(minsk_centroid.reshape((1, minsk_points.shape[1])))[0]


    def _poinc_to_minsk_multi(self, points):
        """
        Converts points from the d-dimensional Poincaré ball model to (d+1)-dimensional Minkowski space.
        Parameters: points (np.ndarray): (N, d) array of points in the Poincaré ball.
        Returns: np.ndarray: (N, d+1) array of points in Minkowski space.
        """
        if isinstance(points, torch.Tensor):  # Ensure input is a NumPy array
            points = points.detach().cpu().numpy()

        norm_sq = np.sum(points**2, axis=1, keepdims=True)  # Compute squared norm of each point
        factor = 2 / (1 - norm_sq)  # Compute scaling factor

        minsk_points = np.zeros((points.shape[0], points.shape[1] + 1))  # Create Minkowski space array
        minsk_points[:, :-1] = factor * points  # First d coordinates
        minsk_points[:, -1] = (1 + norm_sq[:, 0]) / (1 - norm_sq[:, 0])  # Last coordinate

        return minsk_points

    def _minsk_to_poinc_multi(self, points):
        """
        Converts points from (d+1)-dimensional Minkowski space to the d-dimensional Poincaré ball.
        Parameters: pints (np.ndarray): (N, d+1) array of points in Minkowski space.
        Returns: np.ndarray: (N, d) array of points in the Poincaré ball.
        """
        if isinstance(points, torch.Tensor):  # Ensure input is a NumPy array
            points = points.detach().cpu().numpy()
        poinc_points = points[:, :-1] / (1 + points[:, -1][:, None])  # Normalize using last coordinate
        return poinc_points

    
    def predict(self, X):
        """Assign new data points to the nearest centroid."""
        X = X.to(torch.float32)
        distances = torch.stack([self.poincare_distance(X, c) for c in self.centroids])
        return torch.argmin(distances, dim=0)



n_samples = 100
dim = 5

# Initialize Poincaré ball manifold
manifold = geoopt.PoincareBall()

# Generate random points in the Poincaré disk
X = manifold.expmap0(torch.randn(n_samples, dim) * 0.1)  # Small perturbations

# Apply Hyperbolic K-Means
kmeans = HyperbolicKMeans(n_clusters=3, max_iter=50, manifold=manifold)
centroids, labels = kmeans.fit(X)

print(f'vector:{X}')
print(f'centroids:{centroids}')
# Print cluster assignments
print("Cluster assignments:", labels)

label_d = defaultdict(int)
for l in labels:
    lab = l.item()
    label_d[lab] += 1
print(label_d)