import numpy as np
from cloud import *


if __name__ == "__main__":
    data = np.load("galaxies.npy")
    data[:, :-1] = np.deg2rad(data[:, :-1])
    coord_data = np.empty_like(data)
    coord_data[:, 0] = np.multiply(np.sin(data[:, 0]), data[:, 2])
    coord_data[:, 1] = np.multiply(np.cos(data[:, 1]), data[:, 2])
    coord_data[:, 2] = data[:, 2]

    print(np.min(coord_data, axis=0))
    print(np.max(coord_data, axis=0))

    pc = PointCloud(coord_data)

    t1 = time()

    cc = pc.color_cloud(radius=0.01,
                        n_samples=1000,
                        homology_dimension=2,
                        sigma=0.005,
                        n_bins=25,
                        weight_function=lambda x: x**2.5)

    t2 = time()

    print("Total time: " + str(t2 - t1) + "seconds.")

    cc.visualize()