# Similar to vertically_integrated_from_ERA5.py but on the midway server
#%%
import numpy as np
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
g_earth = 9.80665 # Gravity in m/s² as in the IFS documentation

#%%

timestep = '2005-06-02 01:00'

data_path = '/project2/tas1/katharinah/ERA5/hourly_rad_heat/'

# Read vertically integrated MSE
ds_vint = xr.open_dataset(data_path+'era5_vertical_integral_fc_20050601',engine='cfgrib')

# Compute MSE
bud = mseb.mse_budget(config_file='../mse_budget.ini', timestep=timestep, time_interval='1 hour')

#%%
# Vertically integrated MSE
mse_vint = bud.compute_vertical_integral(bud.mse_current)

#%%
ds_ps = xr.open_dataset('/project2/tas1/katharinah/ERA5/ml/2005-06-01/era5_ml_2d_20050601',engine='cfgrib')
surface_pressure = np.exp(np.float64(ds_ps.lnsp.sel(time='2005-06-01 18:00').isel(step=7).values))

#%%
# MSE on model levels
mse_ml = cp * bud.t_current_ml + lh * bud.q_current_ml + bud.geopot_current_ml

# Pressure differences on model levels
p_half = bud.bb[:,np.newaxis,np.newaxis] * surface_pressure[np.newaxis,:,:] + bud.aa[:,np.newaxis,np.newaxis]
p_diff = p_half[1:,:,:] - p_half[:-1,:,:]  # in Pa

# Integrate
mse_vint_ml = np.sum(mse_ml * p_diff, axis=0) / g_earth


#%%
# Integrate the MSE only up to the surface pressure
# But this is not necessary because the values below the surface are set to NaN

pressure_3d = np.zeros_like(bud.mse_current)
pressure_3d[:,:,:] = bud.plev[:,np.newaxis,np.newaxis]

above_surface = np.zeros_like(bud.mse_current)
above_surface[pressure_3d < surface_pressure/100.0] = 1

mse_vint_above_surface = bud.compute_vertical_integral(bud.mse_current * above_surface)

#%%
# Read file with vertically integrated temperature
ds_tvint = xr.open_dataset(data_path+'era5_more_vertical_integral_fc_20050601',engine='cfgrib')
# Plot vertical integral of temperature
ds_tvint.vit.sel(time='2005-06-01 18:00').isel(step=7).plot()

# Compute vertical integral of temperature
t_vint = np.sum(bud.t_current_ml * p_diff, axis=0) / g_earth

#%%
# Total column water vapor
q_vint = np.sum(bud.q_current_ml * p_diff, axis=0) / g_earth
(ds_tvint.tcwv.sel(time='2005-06-01 18:00').isel(step=7) - q_vint).plot(robust=True)

# Geopotential
phi_vint = np.sum(bud.geopot_current_ml * p_diff, axis=0) / g_earth

# Subtract internal energy from potential+internal energy
pe_vint = ds_tvint.vipie.sel(time='2005-06-01 18:00').isel(step=7) - cp * ds_tvint.vit.sel(time='2005-06-01 18:00').isel(step=7)

#%%
(ds_vint.vipile.sel(time='2005-06-01 18:00').isel(step=7) -\
  cp * ds_tvint.vit.sel(time='2005-06-01 18:00').isel(step=7) -\
  lh * ds_tvint.tcwv.sel(time='2005-06-01 18:00').isel(step=7) - phi_vint).plot(robust=True)

#%%
# Read geopotential from 37 pressure level data
ds_geopot = xr.open_dataset('/project2/tas1/abacus/data1/tas/archive/Reanalysis/ERA5/z/era5_z_2005_06.6hrly.nc')


