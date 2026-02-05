# Look at vertically integrated quantities that are downloaded from ERA5 directly.
#%%
import numpy as np
import os
# Do not use reference to my eccodes installation in ~/source_builds.
# Probably not necessary on your system.
os.environ.pop('ECCODES_DIR', None)
os.environ.pop('ECCODES_DEFINITION_PATH', None)

import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import cfgrib

import sys
sys.path.append('..')
import mse_budget_centered_diff as mseb

R_D = 287.0597 # Gas constant for dry air in J/kg/K as in the IFS documentation
cp = 3.5 * R_D # J/kg/K
lh = 2.5008e6 # Latent heat of vaporization in J/kg as in the IFS documentation

data_path = '/media/katharina/Volume/UChicago/ERA5/hourly_rad_heat/'

# Read vertically integrated data from ERA5
ds_vint = xr.open_dataset(data_path+'era5_vertical_integral_fc_20050601',engine='cfgrib')

timestep1 = pd.Timestamp('2005-06-02 01:00')
timestep2 = pd.Timestamp('2005-06-02 02:00')

init_time1, step1 = mseb.get_forecast_time(timestep1)
init_time2, step2 = mseb.get_forecast_time(timestep2)

# Read MSE tendencies
mse_out_path = '/media/katharina/Volume/UChicago/ERA5/mse_output/'
ds_latlon = xr.open_mfdataset(mse_out_path+'with_time_coordinate/*latlon*.nc')

#%%
vminmax = 5000
# Vertical integral of potential + internal + latent energy = MSE
mse1 = ds_vint.vipile.sel(time=init_time1, step=step1)
mse2 = ds_vint.vipile.sel(time=init_time2, step=step2)
mse_tend = (mse2 - mse1) / 3600.0  # W/m2
mse_tend.plot(vmin=-vminmax,vmax=vminmax)
plt.show()
# Plot the MSE tendency that I computed
ds_latlon.mse_tendency.sel(time='2005-06-02 02:00').plot(vmin=-vminmax,vmax=vminmax)
plt.show()

#%%
# Plot the difference
(mse_tend - ds_latlon.mse_tendency.sel(time='2005-06-02 01:00')).plot(vmin=-200,vmax=200)
plt.show()

#%%
# Check whether the difference balances out in the diurnal cycle
ds_latlon_dailymean = ds_latlon.sel(time=slice('2005-06-02 01:00','2005-06-03 0:00')).mean(dim='time')
init_start, step_start = mseb.get_forecast_time(pd.Timestamp('2005-06-02 00:00'))
init_end, step_end = mseb.get_forecast_time(pd.Timestamp('2005-06-03 00:00'))
mse_start = ds_vint.vipile.sel(time=init_start, step=step_start)
mse_end = ds_vint.vipile.sel(time=init_end, step=step_end)
mse_tend_daily = (mse_end - mse_start) / (24 * 3600.0)  # W/m2
(mse_tend_daily - ds_latlon_dailymean.mse_tendency).plot(robust=True)
# Difference is now mainly in extratropical cyclones, wavenumber 2 structure is gone.

#%%
# MSE flux divergence
thermal_fluxdiv = ds_vint.vithed.sel(time=init_time1, step=step1)
moisture_fluxdiv = ds_vint.vimdf.sel(time=init_time1, step=step1)
geopot_fluxdiv = ds_vint.vigd.sel(time=init_time1, step=step1)
mse_fluxdiv = thermal_fluxdiv + lh * moisture_fluxdiv + geopot_fluxdiv

# MSE advection from my calculation
total_advection = (ds_latlon.zonal_adv_mse + ds_latlon.meridional_adv_mse + ds_latlon.vertical_adv_mse)

# As for the MSE tendency, the difference has a wavenumber 2 structure
(mse_fluxdiv - total_advection.sel(time=timestep1)).plot(vmin=-200,vmax=200)
plt.show()

#%%
# Does the MSE budget close when comparing it with the radiative and heat fluxes?


