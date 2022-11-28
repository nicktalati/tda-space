from gtda.homology import VietorisRipsPersistence as VRP
import open3d as otd
import numpy as np

np.random.seed(42)
np.set_printoptions(threshold=np.inf)


class Cloud:
    """
    This class holds point-cloud data and has methods to visualize
    the cloud, create persistence diagrams, and create subclouds. 
    """
    def __init__(self, points = []):
        self.points = np.asarray(points, dtype=float)
        self.num_points = self.points.shape[0]
    
    def __getitem__(self, index):
        return self.points[index]

    def __setitem__(self, index, value):
        self.points[index] = np.asarray(value, dtype=float)

    def __len__(self):
        return self.num_points

    def __str__(self) -> str:
        return str(self.points)

    def append(self, value):
        value = np.asarray(value, dtype=float)
        value = np.expand_dims(value, 0)
        self.points = np.concatenate([self.points, value], axis=0)
        self.num_points += 1

    def persistence_diagram(self, homology_dimensions=[0]):
        """
        Returns a persistence diagram for the point cloud. The
        persistence diagram is an ndarray with shape (n, 3), where
        n is the number of points in the cloud and each of the 3
        dimensions gives the birth time, death time, and homology
        dimension, respectively. 
        """
        VR = VRP(homology_dimensions=homology_dimensions)
        return VR.fit_transform([self.points])[0]

    def visualize(self, colors=None):
        """
        Opens the cloud in a new window for visualization. The colors
        parameter has shape (n, 3), where n is the number of points in the
        cloud and each entry is an rgb value in [0, 1].
        """
        if colors is None:
            colors = np.zeros_like(self.points)
        colors = np.asarray(colors)
        pc = otd.geometry.PointCloud()
        pc.points = otd.utility.Vector3dVector(self.points)
        pc.colors = otd.utility.Vector3dVector(colors)
        otd.visualization.draw_geometries([pc])

    def subcloud(self, center, radius):
        center = np.asarray(center, dtype=float)
        center_array = np.tile(center, (self.num_points, 1))
        diffs = center_array - self.points
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        points = []
        for i in range(self.num_points):
            if distances[i] <= radius:
                points.append(self.points[i])
        points = np.asarray(points, dtype=float)
        return Cloud(points)


def make_torus():
    cloud = []
    for x in np.linspace(-1.25, 1.25, 100):
        for y in np.linspace(-1.25, 1.25, 100):
            for z in np.linspace(-1.25, 1.25, 100):
                if (np.abs((0.5 - np.sqrt(x**2 + y**2))**2 + z**2 - 0.25**2) < 0.002):
                    cloud.append([x, y, z])
    cloud = np.asarray(cloud)
    cloud += 0.1 * np.random.random(cloud.shape)
    return Cloud(np.asarray(cloud))


if __name__ == "__main__":
    torus = make_torus()
    colors = np.random.random(torus.points.shape)

    torus.visualize(colors)
    subcloud = torus.subcloud([0.5,0,0], 0.5)
    subcloud.visualize()

