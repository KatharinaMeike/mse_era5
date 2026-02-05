# Compare the MSE tendencies and fluxes for 37 pressure levels between 
# the data that I have interpolated, and the data interpolated by the ECMWF.
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
    

ds_me = xr.open_dataset('/media/katharina/Volume/UChicago/ERA5/mse_output/mse_daily_P37_latlon_2005060300.nc')
ds_fromERA5 = xr.open_dataset('/media/katharina/Volume/UChicago/ERA5/mse_output/mse_daily_P37_varp_latlon_2005060300.nc')

ds_me_levlat = xr.open_dataset('/media/katharina/Volume/UChicago/ERA5/mse_output/mse_daily_P37_levlat_2005060300.nc')
ds_fromERA5_levlat = xr.open_dataset('/media/katharina/Volume/UChicago/ERA5/mse_output/mse_daily_P37_varp_levlat_2005060300.nc')

fig, axes = plt.subplots(2,4,figsize=(12,7))
axes = axes.flatten()
# MSE tendency: does not match because of the data assimilation.
ax = axes[0]
ds_me.mse_tendency.mean(dim='longitude').plot(ax=ax,y='latitude',label='I interpolated',color='blue')
ds_fromERA5.mse_tendency.mean(dim='longitude').plot(ax=ax,y='latitude',label='P37 standard',color='red',linestyle='dashed')
ax.set_title('MSE Tendency')
ax.set_ylabel('')
ax.legend()


ax = axes[1]
ds_me.zonal_adv_mse.mean(dim='longitude').plot(ax=ax,y='latitude',color='blue')
ds_fromERA5.zonal_adv_mse.mean(dim='longitude').plot(ax=ax,y='latitude',color='red',linestyle='dashed')
ax.set_title('Zonal MSE Advection')
ax.set_xlabel('W m$^{-2}$')
ax.set_ylabel('')

ax = axes[2]
ds_me.meridional_adv_mse.mean(dim='longitude').plot(ax=ax,y='latitude',color='blue')
ds_fromERA5.meridional_adv_mse.mean(dim='longitude').plot(ax=ax,y='latitude',color='red',linestyle='dashed')
ax.set_title('Meridional MSE Advection')
ax.set_xlabel('W m$^{-2}$')
ax.set_ylabel('')

ax = axes[3]
ds_me.vertical_adv_mse.mean(dim='longitude').plot(ax=ax,y='latitude',color='blue')
ds_fromERA5.vertical_adv_mse.mean(dim='longitude').plot(ax=ax,y='latitude',color='red',linestyle='dashed')
ax.set_title('Vertical MSE Advection')
ax.set_xlabel('W m$^{-2}$')
ax.set_ylabel('')

"""mse_advection = -(ds.zonal_adv_mse.mean(dim='longitude') +\
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
ax.axvline(0,color='gray')"""

plt.subplots_adjust(wspace=0.4, hspace=0.4)
#plt.savefig(plot_path+f'daily_vs_hourly_vint_mean_{plot_timestring}.png',dpi=300,bbox_inches='tight')

#%%
variable_names = ['mse_tendency','zonal_adv_mse','meridional_adv_mse','vertical_adv_mse',\
                  'zonal_mse_flux','meridional_mse_flux','vertical_mse_flux']
vminmax_list = [0.05,0.1,0.1,0.5,5e7,1e6,2e4]

for variable_name, vminmax in zip(variable_names[6:],vminmax_list[6:]):
    fig = plt.figure(figsize=(12,7))
    gs = gridspec.GridSpec(3,2, wspace=0.2,hspace=0.35)
    ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
    cs = ds_me[variable_name].plot(ax=ax, x='longitude', y='latitude', cmap='RdBu_r',\
                        center=0,transform=ccrs.PlateCarree())
    ax.coastlines(color='gray')
    ax.set_title(variable_name + ' (I interpolated)')

    ax = fig.add_subplot(gs[1,0],projection=ccrs.PlateCarree())
    cs = ds_fromERA5[variable_name].plot(ax=ax, x='longitude', y='latitude', cmap='RdBu_r',\
                        center=0,transform=ccrs.PlateCarree())
    ax.coastlines(color='gray')
    ax.set_title(variable_name + ' (P37 standard)')

    ax = fig.add_subplot(gs[2,0],projection=ccrs.PlateCarree())
    cs = (ds_me[variable_name] - ds_fromERA5[variable_name]).plot(ax=ax, x='longitude', y='latitude', cmap='RdBu_r',\
                        center=0,transform=ccrs.PlateCarree())
    ax.coastlines(color='gray')
    ax.set_title('Difference')

    ax = fig.add_subplot(gs[0,1])
    ds_me_levlat[variable_name].plot(ax=ax,vmin=-vminmax,vmax=vminmax,cmap='RdBu_r')
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_title(variable_name + ' (I interpolated)')
    ax.set_xlabel('')

    ax = fig.add_subplot(gs[1,1])
    ds_fromERA5_levlat[variable_name].plot(ax=ax,vmin=-vminmax,vmax=vminmax,cmap='RdBu_r')
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_title(variable_name + ' (P37 standard)')
    ax.set_xlabel('')
    ax = fig.add_subplot(gs[2,1])
    (ds_me_levlat[variable_name] - ds_fromERA5_levlat[variable_name]).plot(ax=ax,vmin=-vminmax,vmax=vminmax,cmap='RdBu_r')
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_xlabel('')
    plt.show()