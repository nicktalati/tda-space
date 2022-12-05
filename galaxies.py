import numpy as np
from cloud import *


if __name__ == "__main__":
    data = np.load("simulated-data.npy")
    colors = np.zeros_like(data)
    data = data / 2399
    data = data - 0.5 * np.ones_like(data)
    data = rotate(data, 0, 1)

    pc = PointCloud(data)
    pc.visualize()

    t1 = time()

    cc = pc.color_cloud(radius=0.1,
                        n_samples=50,
                        n_points=100,
                        homology_dimension=1,
                        sigma=0.005,
                        n_bins=25,
                        weight_function=lambda x: x**3)

    t2 = time()

    print("Total time: " + str(t2 - t1) + "seconds.")

    plt.imsave("examples/cosmic_slice.png", cc.colors)

    cc.visualize(save_folder="pngs")