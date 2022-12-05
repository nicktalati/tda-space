import numpy as np
from astropy.io import fits


data = np.loadtxt("all-galaxy.txt")
print(data.shape)
np.save("galaxies.npy", data)

"""

data = fits.open('data3.fits')
data = data[0].data

print(data.shape)

ras = data["objra"]
decs = data["objdec"]
distances = data["nsa_zdist"]

data = np.vstack([ras, decs, distances]).transpose()
#np.save("data.npy", data)
"""
