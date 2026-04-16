import netCDF4
import matplotlib.pyplot as plt

fb = "data/sst.mon.ltm.1981-2010.nc"
nc = netCDF4.Dataset(fb)

print(nc)