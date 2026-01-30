# Plot the output of save_fluxdiv_short_time.py and compare with 
# radiative quantities and precipitation

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import cfgrib
import mse_budget_centered_diff as mseb

data_path = '/project2/tas1/katharinah/ERA5/mse_output/'
era5_path = '/project2/tas1/abacus/data1/tas/archive/Reanalysis/ERA5/'

timestep = pd.Timestamp('2005-06-02 08:00')

ds_levlat = xr.open_dataset(data_path + 'mse_tendency_levlat.nc' + timestep.strftime('_%Y%m%d%H') + '.nc') 
ds_latlon = xr.open_dataset(data_path + 'mse_tendency_latlon.nc' + timestep.strftime('_%Y%m%d%H') + '.nc') 


# Monthly data
# Surface heat fluxes
ds_stf = xr.open_dataset(era5_path+'era5_stf_1979_2019.nc').sel(time='2005-06-01')

# Radiative quantities
ds_rad = xr.open_dataset(era5_path+'era5_rad_1979_2019.nc').sel(time='2005-06-01')
ra = ds_rad.tsr + ds_rad.ttr - ds_rad.ssr - ds_rad.str


#%%
# Hourly radiation and surface heat fluxes
ds_hourly = xr.open_dataset('/project2/tas1/katharinah/ERA5/hourly_rad_heat/era5_single_level_analysis_20050601',engine='cfgrib')

time_fc, step_fc = mseb.get_forecast_time(timestep)

ds_rt = ds_hourly.sel(time=time_fc, step=step_fc)
ra_hourly = ds_rt.tsr + ds_rt.ttr - ds_rt.ssr - ds_rt.str # out of the atmosphere
shf_hourly = ds_rt.sshf
lhf_hourly = ds_rt.slhf

#%%
fig, ax = plt.subplots(figsize=(5,3))
(ds_latlon.mse_tendency.mean(dim='longitude')*3600.0).plot(ax=ax,y='latitude',label='MSE tendency',color='k')
(ds_latlon.dy_meridional_mse_flux.mean(dim='longitude')*3600.0).plot(ax=ax,y='latitude',label='Meridional MSE flux divergence',color='b')
ra_hourly.mean(dim='longitude').plot(ax=ax,y='latitude',label='Radiative cooling',color='r')
shf_hourly.mean(dim='longitude').plot(ax=ax,y='latitude',label='Surface heat flux',color='g')
lhf_hourly.mean(dim='longitude').plot(ax=ax,y='latitude',label='Latent heat flux',color='orange')
ax.set_xlim(-1e6,1e6)

#%%
#%%
mse_advection =  3600.0 * (ds_latlon.zonal_adv_mse.mean(dim='longitude') +\
                ds_latlon.meridional_adv_mse.mean(dim='longitude') +\
                ds_latlon.vertical_adv_mse.mean(dim='longitude'))

dy_meridional_mse_flux = ds_latlon.dy_meridional_mse_flux.mean(dim='longitude')*3600.0

div_mse_flux = (ds_latlon.dy_meridional_mse_flux.mean(dim='longitude') +\
                          ds_latlon.dx_zonal_mse_flux.mean(dim='longitude') +\
                          ds_latlon.dp_vertical_mse_flux.mean(dim='longitude'))*3600.0

tendency = ds_latlon.mse_tendency.mean(dim='longitude')*3600.0

residual = - ra_hourly.mean(dim='longitude') + shf_hourly.mean(dim='longitude') +\
      lhf_hourly.mean(dim='longitude') + mse_advection + tendency
inferred_merid = ra_hourly.mean(dim='longitude') - shf_hourly.mean(dim='longitude') -\
      lhf_hourly.mean(dim='longitude') - tendency

single_level = - ra_hourly.mean(dim='longitude') + shf_hourly.mean(dim='longitude') +\
      lhf_hourly.mean(dim='longitude') 


fig, ax = plt.subplots(figsize=(5,5))
tendency.plot(ax=ax,y='latitude',label='MSE tendency',color='k')
mse_advection.plot(ax=ax,y='latitude',label='Meridional MSE flux divergence',color='b')
(-1.0 * ra_hourly.mean(dim='longitude')).plot(ax=ax,y='latitude',label='Radiative cooling',color='r')
shf_hourly.mean(dim='longitude').plot(ax=ax,y='latitude',label='Surface heat flux',color='g')
lhf_hourly.mean(dim='longitude').plot(ax=ax,y='latitude',label='Latent heat flux',color='orange')

inferred_merid.plot(ax=ax,y='latitude',label='Residual',color='purple',linestyle='--')

dy_meridional_mse_flux.plot(ax=ax,y='latitude',label='Meridional MSE flux divergence',color='cyan')
div_mse_flux.plot(ax=ax,y='latitude',label='Total MSE flux divergence',color='blue',linestyle='--')

residual.plot(ax=ax,y='latitude',label='Residual',color='magenta')
ax.set_xlim(-1e6,1e6)
ax.axvline(0.0,color='gray',linestyle=':')

