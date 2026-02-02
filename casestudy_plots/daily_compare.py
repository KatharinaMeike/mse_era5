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

import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

sys.path.append('..')

import mse_budget_centered_diff as mseb

timestep = pd.Timestamp('2005-06-02 01:00')

average_hours = 23 # If >0, average over timestep + average_hours hours

data_path = '/media/katharina/Volume/UChicago/ERA5/mse_output/'
plot_path = '/media/katharina/Volume/UChicago/MSE/plots/casestudy/'

# Read MSE advection, tendencies and fluxes
ds_latlon = xr.open_mfdataset(data_path+'with_time_coordinate/*latlon*.nc')
if average_hours > 0:
    ds = ds_latlon.sel(time=slice(timestep,timestep + pd.Timedelta(hours=average_hours))).mean(dim='time')
    plot_timestring = f'{timestep:%Y-%m-%d %H} to {timestep + pd.Timedelta(hours=average_hours):%Y-%m-%d %H} UTC'
else:
    ds = ds_latlon.sel(time=timestep)
    plot_timestring = f'{timestep:%Y-%m-%d %H} UTC'
    

ds_daily = xr.open_dataset('/media/katharina/Volume/UChicago/ERA5/mse_output/mse_daily_P137_latlon_2005060300.nc')

fig, axes = plt.subplots(2,4,figsize=(12,7))
axes = axes.flatten()
# MSE tendency: does not match because of the data assimilation.
ax = axes[0]
ds_daily.mse_tendency.mean(dim='longitude').plot(ax=ax,y='latitude',label='Daily Data',color='blue')
ds.mse_tendency.mean(dim='longitude').plot(ax=ax,y='latitude',label='Hourly Data',color='red',linestyle='dashed')
ax.set_title('MSE Tendency')
ax.set_ylabel('')
ax.legend()

ax = axes[1]
ds_daily.mse_tendency_convection.mean(dim='longitude').plot(ax=ax,y='latitude',label='Daily Data',color='blue')
ds.mse_tendency_convection.mean(dim='longitude').plot(ax=ax,y='latitude',label='Hourly Data',color='red',linestyle='dashed')
ax.set_title('MSE Tendency due to Convection')
ax.set_xlabel('W m$^{-2}$')
ax.set_ylabel('')

ax = axes[2]
ds_daily.mse_tendency_shortwave.mean(dim='longitude').plot(ax=ax,y='latitude',label='Daily Data',color='blue')
ds.mse_tendency_shortwave.mean(dim='longitude').plot(ax=ax,y='latitude',label='Hourly Data',color='red',linestyle='dashed')
ax.set_title('Shortwave Radiation')
ax.set_xlabel('W m$^{-2}$')
ax.set_ylabel('')

ax = axes[3]
ds_daily.mse_tendency_longwave.mean(dim='longitude').plot(ax=ax,y='latitude',label='Daily Data',color='blue')
ds.mse_tendency_longwave.mean(dim='longitude').plot(ax=ax,y='latitude',label='Hourly Data',color='red',linestyle='dashed')
ax.set_title('Longwave Radiation')
ax.set_xlabel('W m$^{-2}$')
ax.set_ylabel('')

ax = axes[4]
ds_daily.zonal_adv_mse.mean(dim='longitude').plot(ax=ax,y='latitude',label='Daily Data',color='blue')
ds.zonal_adv_mse.mean(dim='longitude').plot(ax=ax,y='latitude',label='Hourly Data',color='red',linestyle='dashed')
ax.set_title('Zonal MSE Advection')
ax.set_xlabel('W m$^{-2}$')
ax.set_ylabel('')

ax = axes[5]
ds_daily.meridional_adv_mse.mean(dim='longitude').plot(ax=ax,y='latitude',label='Daily Data',color='blue')
ds.meridional_adv_mse.mean(dim='longitude').plot(ax=ax,y='latitude',label='Hourly Data',color='red',linestyle='dashed')
ax.set_title('Meridional MSE Advection')
ax.set_xlabel('W m$^{-2}$')
ax.set_ylabel('')

ax = axes[6]
ds_daily.vertical_adv_mse.mean(dim='longitude').plot(ax=ax,y='latitude',label='Daily Data',color='blue')
ds.vertical_adv_mse.mean(dim='longitude').plot(ax=ax,y='latitude',label='Hourly Data',color='red',linestyle='dashed')
ax.set_title('Vertical MSE Advection')
ax.set_xlabel('W m$^{-2}$')
ax.set_ylabel('')

mse_advection = -(ds.zonal_adv_mse.mean(dim='longitude') +\
                 ds.meridional_adv_mse.mean(dim='longitude') +\
                 ds.vertical_adv_mse.mean(dim='longitude'))

tendency = ds.mse_tendency.mean(dim='longitude')

residual_hourly = mse_advection - tendency + ds.mse_tendency_convection.mean(dim='longitude') +\
      + ds.mse_tendency_shortwave.mean(dim='longitude') + ds.mse_tendency_longwave.mean(dim='longitude')

residual_daily = -(ds_daily.zonal_adv_mse.mean(dim='longitude') +\
                 ds_daily.meridional_adv_mse.mean(dim='longitude') +\
                 ds_daily.vertical_adv_mse.mean(dim='longitude')) -\
      ds_daily.mse_tendency.mean(dim='longitude') + ds_daily.mse_tendency_convection.mean(dim='longitude') +\
      + ds_daily.mse_tendency_shortwave.mean(dim='longitude') + ds_daily.mse_tendency_longwave.mean(dim='longitude')

ax = axes[7]
residual_daily.plot(ax=ax,y='latitude',label='Daily Data',color='blue')
residual_hourly.plot(ax=ax,y='latitude',label='Hourly Data',color='red',linestyle='dashed')
ax.set_title('Residual')
ax.set_xlabel('W m$^{-2}$')
ax.set_ylabel('')
ax.axvline(0,color='gray')

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(plot_path+f'daily_vs_hourly_vint_mean_{plot_timestring}.png',dpi=300,bbox_inches='tight')