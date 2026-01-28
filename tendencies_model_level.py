# Read model-level data, downloaded from ECMWF
#%%

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

data_path = '/project2/tas1/katharinah/ERA5/ml/'

#ds_uv = xr.open_dataset(data_path+'era5_model_level_wind.nc')

#ds = xr.open_dataset(data_path+'era5_model_level.nc')
ds = xr.open_dataset(data_path+'era5_model_level_var.nc')


