# Rename the coordinate 'valid_time' to 'time' in the vertical velocity data.
#%%
import numpy as np
import xarray as xr

data_path = '/project2/tas1/katharinah/ERA5/w/'

ds = xr.open_dataset(data_path+'era5_pl_w_200506.nc')

ds_renamed = ds.rename({'valid_time':'time','pressure_level':'level'})

ds_renamed.to_netcdf(data_path+'era5_pl_w_200506_renamed.nc')


