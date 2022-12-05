import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from cloud import *


data = fits.open('gas.fits')
data = data[0].data[0]
data = data / np.sum(data)

distribution = np.reshape(data, (data.shape[0] * data.shape[1]))
print(distribution.shape)

cloud = []

t1 = time()
points = np.random.choice(np.arange(5760000), size=(100000), p=distribution)
t2 = time()

print("total time:", t2 - t1)
for point in points:
    x = point % 2400
    y = int(point / 2400)
    cloud.append([x, y, 0])

np.save("simulated-data.npy", np.asarray(cloud))

pc = PointCloud(np.asarray(cloud))
pc.visualize()
