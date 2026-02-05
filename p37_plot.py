# Plot MSE tendency, advection and fluxes, computed with p37_mse_budget.py
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

#%%

ds = xr.open_mfdataset('/project2/tas1/katharinah/ERA5/mse_output/with_time_coordinate/mse_daily_P37_varp_latlon_200501*.nc')

ds.mean(dim='time').to_netcdf('/project2/tas1/katharinah/ERA5/mse_output/monmean/mse_daily_P37_varp_latlon_20050102_20050131.nc')
#ds = xr.open_mfdataset('/project2/tas1/katharinah/ERA5/mse_output/mse_daily_P37_varp_latlon_200503*.nc')

ds = xr.open_mfdataset('/project2/tas1/katharinah/ERA5/mse_output/with_time_coordinate/mse_daily_P37_varp_levlat_200501*.nc')

ds.mean(dim='time').to_netcdf('/project2/tas1/katharinah/ERA5/mse_output/monmean/mse_daily_P37_varp_levlat_20050102_20050131.nc')

#%%
import cfgrib
# Check if the tendencies dataset has all variables
ds_tend = xr.open_dataset('/project2/tas1/katharinah/ERA5/ml/2005-06-04/era5_ml_tend_20050604',engine='cfgrib',\
            filter_by_keys={'typeOfLevel':'hybrid'})

#%%
ds_surface = xr.open_dataset('/project2/tas1/katharinah/ERA5/ml/2005-06-04/era5_ml_tend_20050604',engine='cfgrib',\
            filter_by_keys={'shortName':'z'})

#%%
ds_surface = xr.open_dataset('/project2/tas1/katharinah/ERA5/ml/2005-06-04/era5_ml_tend_20050604',engine='cfgrib',\
            filter_by_keys={'shortName':'lnsp'})
