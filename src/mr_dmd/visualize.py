import netCDF4
import matplotlib.pyplot as plt
import numpy as np

fb = "data/sst.mon.mean.nc"
nc = netCDF4.Dataset(fb)
sst = nc["sst"][:, :, :]

# test = np.load("data/sst.npy")

# print(type(test))

# print(test.shape)
print(type(sst))
# flatten dimensions 1 and 2 (combine lat/lon) so result is (time, space)
sst_flat = sst.reshape(sst.shape[0], -1)
print(type(sst_flat))
print(sst_flat.shape)

print("mean at start")
print(np.mean(sst[0, :, :]))

print("mean at end")
print(np.mean(sst[-1, :, :]))
