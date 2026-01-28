# Compute MSE and possibly its tendencies for two timesteps in ERA5 data
#%%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

dry_R = 287.0 # Specific gas constant
cp = 3.5 * dry_R # J/kg/K
lh = 2264705.0 # J/kg


data_path = '/project2/tas1/abacus/data1/tas/archive/Reanalysis/ERA5/'

# Temperature
ds_t = xr.open_dataset(data_path+'t/era5_t_2020_07.6hrly.nc')

# Specific humidity
ds_q = xr.open_dataset(data_path+'q/era5_q_2020_07.6hrly.nc')

# Geopotential
ds_z = xr.open_dataset(data_path+'z/era5_z_2020_07.6hrly.nc')

#%%
# Compute moist static energy for selected time
timestep = pd.to_datetime('2020-07-01 0:00')
mse = (ds_t.t.sel(time=timestep) * cp + ds_q.q.sel(time=timestep) * lh + ds_z.z.sel(time=timestep)).rename('mse')
mse.attrs['long_name'] = 'moist static energy'

#%%
# Plot vertical profile
fig, ax = plt.subplots()
for lon in range(0,360,30):
    mse.sel(level=slice(100,1000)).sel(latitude=0,longitude=lon).plot(ax=ax,y='level')
ax.invert_yaxis()

