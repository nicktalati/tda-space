from gtda.homology import VietorisRipsPersistence as VRP
from gtda.diagrams import PersistenceImage
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import open3d as otd
import numpy as np

from time import time

np.random.seed(42)
np.set_printoptions(threshold=np.inf)


def normalize(arr):
    """
    Normalizes columns of arr to be in [0, 1].
    """
    ret = np.asarray(arr)
    ret = ret - ret.mean(axis=0)
    std = ret.std(axis=0)
    std += np.ones_like(std) * (std == 0)
    ret = ret / (2 * std)
    ret += 0.5
    ret = np.minimum(np.ones_like(ret), ret)
    ret = np.maximum(np.zeros_like(ret), ret)
    return ret


class PointCloud(np.ndarray):
    """
    Class for storing np.ndarrays as point clouds. 
    """
    def __new__(cls, input_array, colors=None):
        obj = np.asarray(input_array).view(cls)
        obj.n_points = len(input_array)
        obj.subclouds = {}
        obj.colors = colors
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.n_points = getattr(obj, 'info', None)
        self.subclouds = getattr(obj, 'subclouds', None)
        self.colors = getattr(obj, "colors", None)

    def persistence_diagram(self, homology_dims=[0]):
        """
        Returns a persistence diagram for the point cloud. The 
        diagram is an ndarray of triples; the first entry is the birth
        time, second is the death time, third is the homoloy dimension.
        """
        vrp_transformer = VRP(homology_dimensions=homology_dims)
        return vrp_transformer.fit_transform([self])

    def persistence_image(self, homology_dims=[0], sigma=0.005, n_bins=10,
                          weight_function=lambda x: x**2):
        """
        Returns a persistence image for the point cloud. Sigma gives the size
        of the gaussian kernel used for smoothing the image. The image is of
        size n_bins x n_bins. The weight function scales each feature by a 
        function of x, where x = death - birth. 
        """
        pd = self.persistence_diagram(homology_dims=homology_dims)
        pi_transformer = PersistenceImage(sigma=sigma,
                                          n_bins=n_bins,
                                          weight_function=weight_function)
        persistence_image = pi_transformer.fit_transform(pd)
        return persistence_image

    def visualize(self):
        """
        Opens the cloud in a new window for visualization.
        """
        colors = self.colors
        if colors is None:
            colors = np.zeros_like(self)
        pc = otd.geometry.PointCloud()
        pc.points = otd.utility.Vector3dVector(self)
        pc.colors = otd.utility.Vector3dVector(colors)
        otd.visualization.draw_geometries([pc])

    def subcloud(self, center, radius):
        """
        Returns a subcloud of self containing the points that are
        inside the sphere with centered at center with radius radius.
        """
        if (tuple(center), radius) in self.subclouds.keys():
            return self.subclouds[(tuple(center), radius)] 

        center = np.asarray(center, dtype=float)
        center_array = np.tile(center, (self.n_points, 1))
        diffs = center_array - self
        distances = np.sqrt(np.sum(diffs**2, axis=1))

        points = self[distances <= radius]

        if len(points) < 4:
            return None

        cloud = PointCloud(points)
        self.subclouds[(tuple(center), radius)] = cloud
        return cloud
    
    def make_subclouds(self, radius, n_samples=1000):
        """
        Samples n_samples points in self. For each sampled point, the
        subcloud of radius radius is added to self.subclouds. 
        """
        np.random.shuffle(self)
        mx = len(self)
        for i in range(n_samples):
            i %= mx
            self.subcloud(self[i], radius)

    def color_cloud(self, radius, n_samples, homology_dimension, sigma=0.005,
                    n_bins=10, weight_function=lambda x: x**2):
        """
        This function creates another point cloud from self. First, n_samples
        subclouds are made, each of radius radius. Persistence images are 
        calculated for each one according to the other parameters, described 
        in the persistence_image method. Each persistence image is vectorized,
        then PCA reduces the dimension of each image to 3. The values of the 
        new vector in R^3 correspond to RGB values, after they are normalized.
        These become the color of the center of the sampled subcloud. The 
        method returns the PointCloud whose points are composed of the sampled
        points from self and whose colors correspond to the colors calculated.
        """
        if self.subclouds == {}:
            self.make_subclouds(radius, n_samples=n_samples)
        
        kvs = self.subclouds.items()
        centers = [kv[0][0] for kv in kvs]
        centers = np.asarray(centers)
        subclouds = [np.asarray(kv[1]) for kv in kvs]

        vrp_transformer = VRP(homology_dimensions=[homology_dimension])
        persistence_diagrams = vrp_transformer.fit_transform(subclouds)
        pi_transformer = PersistenceImage(sigma=sigma,
                                          n_bins=n_bins,
                                          weight_function=weight_function)
        persistence_images = pi_transformer.fit_transform(persistence_diagrams)

        pi_shape = persistence_images.shape
        images_linearized = persistence_images.reshape(*pi_shape[:2], -1)
        images_linearized = np.squeeze(images_linearized, axis=1)

        pca_transformer = PCA(3)
        colors = pca_transformer.fit_transform(images_linearized)
        colors = normalize(colors)

        return PointCloud(centers, colors)


def make_torus(r1, r2, noise=0.0, resolution=100):
    """
    Makes a torus PointCloud where the density of any patch of area on the
    surface of the torus is essentially constant. 
    """
    points = []
    for theta in np.linspace(0, 2*np.pi, resolution, endpoint=False):
        radius = r1 - r2 * np.cos(theta)
        height = r2 * np.sin(theta)
        n_points = int(resolution * radius / r1)
        for phi in np.linspace(0, 2*np.pi, n_points, endpoint=False):
            x = radius * np.cos(phi)
            y = radius * np.sin(phi)
            points.append([x, y, height])
    points = np.asarray(points)
    noise_mat = np.random.random(points.shape) + np.full_like(points, -0.5)
    noise_mat *= noise
    points += noise_mat
    return PointCloud(points)


def make_sphere(r, noise=0.0, n_points=1000):
    """
    Makes a sphere PointCloud. 
    """
    vecs = np.random.random((n_points, 3))
    vecs -= np.full_like(vecs, 0.5)
    lens = np.sqrt(np.sum(vecs**2, axis=1))
    lens = np.tile(lens, (3, 1)).transpose()
    vecs = vecs / lens
    vecs *= float(r)
    noise_mat = np.random.random(vecs.shape) + np.full_like(vecs, -0.5)
    noise_mat *= noise
    vecs += noise_mat
    return PointCloud(vecs * float(r))


if __name__ == "__main__":
    SUBCLOUD_RADIUS = 0.3 # radius of samples taken from larger space
    N_SUBCLOUDS = 10000 # number of samples taken
    HOM_DIM = 1 # homological dimension for which to calculate persistence
    SIGMA = 0.005 # size of gaussian filter that smooths persistence images
    N_BINS = 100 # persistence images are of size N_BINS x N_BINS
    WEIGHT_FUNC = lambda x: x ** 2.5 # scales persistence image based on 
                                     # x = death - birth

    torus = make_torus(1, 0.9, noise=0.03, resolution=100)
    torus.visualize()

    cc = torus.color_cloud(SUBCLOUD_RADIUS,
                           N_SUBCLOUDS,
                           HOM_DIM,
                           sigma=SIGMA,
                           n_bins=N_BINS,
                           weight_function=WEIGHT_FUNC)

    cc.visualize()

