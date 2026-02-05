# Also here, the vertical integral of several quantities in the MSE budget is plotted.
# But here, as horizontal maps.
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
elif average_hours == -2: # daily mean, p37
    ds = xr.open_dataset('/media/katharina/Volume/UChicago/ERA5/mse_output/mse_daily_P37_latlon_2005060300.nc')
    plot_timestring = '2005060300_dailymean_P37'
else:
    ds = ds_latlon.sel(time=timestep)
    plot_timestring = f'{timestep:%Y-%m-%d %H} UTC'

# Read TOA and surface fluxes (single-level data)
ds_single = xr.open_dataset('/media/katharina/Volume/UChicago/ERA5/hourly_rad_heat/era5_single_level_fc_20050601',engine='cfgrib')

# Select time in single-level dataset
init_time, step = mseb.get_forecast_time(timestep)
ds_tb = ds_single.sel(time=init_time, step=step)

if average_hours > 0:
    for h in range(1,average_hours+1):
        init_time_h, step_h = mseb.get_forecast_time(timestep + pd.Timedelta(hours=h))
        ds_tb_h = ds_single.sel(time=init_time_h, step=step_h)
        ds_tb = ds_tb + ds_tb_h
    ds_tb = ds_tb / (average_hours + 1)

# Heat fluxes are negative in the mid-latitudes and tropics. 
# They are multiplied by -1 because we are interested in the flux into the atmosphere.
sensible_heat_flux = -ds_tb.avg_ishf 
latent_heat_flux = -ds_tb.avg_slhtf

# Net radiative flux into the atmosphere
ra = ds_tb.avg_tnswrf - ds_tb.avg_snswrf + ds_tb.avg_tnlwrf - ds_tb.avg_snlwrf
shortwave_single = ds_tb.avg_tnswrf - ds_tb.avg_snswrf
longwave_single = ds_tb.avg_tnlwrf - ds_tb.avg_snlwrf

#%%
vminmax = 1000

fig = plt.figure(figsize=(13,12))
gs = gridspec.GridSpec(5, 4, width_ratios=[1,1,1,0.03], wspace=0.05,hspace=0.2)
ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
cs = ds.mse_tendency.plot(ax=ax, x='longitude', y='latitude', cmap='RdBu_r',\
                     vmin=-vminmax,vmax=vminmax,transform=ccrs.PlateCarree(),\
                     add_colorbar=False)
ax.coastlines(color='gray')
ax.set_title('MSE tendency')

total_advection = -(ds.zonal_adv_mse + ds.meridional_adv_mse + ds.vertical_adv_mse)
ax = fig.add_subplot(gs[0,1],projection=ccrs.PlateCarree())
total_advection.plot(ax=ax, x='longitude', y='latitude', cmap='RdBu_r',\
                     vmin=-vminmax,vmax=vminmax,transform=ccrs.PlateCarree(),\
                     add_colorbar=False)
ax.coastlines(color='gray')
ax.set_title('MSE advection')

total_fluxdiv = -(ds.dx_zonal_mse_flux + ds.dy_meridional_mse_flux + ds.dp_vertical_mse_flux)
ax = fig.add_subplot(gs[0,2],projection=ccrs.PlateCarree())
total_fluxdiv.plot(ax=ax, x='longitude', y='latitude', cmap='RdBu_r',\
                     vmin=-vminmax,vmax=vminmax,transform=ccrs.PlateCarree(),\
                     add_colorbar=False)
ax.coastlines(color='gray')
ax.set_title('MSE flux divergence')

ax = fig.add_subplot(gs[1,0],projection=ccrs.PlateCarree())
(-ds.zonal_adv_mse).plot(ax=ax, x='longitude', y='latitude', cmap='RdBu_r',add_colorbar=False,\
                      vmin=-vminmax,vmax=vminmax,transform=ccrs.PlateCarree())
ax.coastlines(color='gray')
ax.set_title('Zonal MSE advection')

ax = fig.add_subplot(gs[1,1],projection=ccrs.PlateCarree())
(-ds.meridional_adv_mse).plot(ax=ax, x='longitude', y='latitude', cmap='RdBu_r',add_colorbar=False,\
                      vmin=-vminmax,vmax=vminmax,transform=ccrs.PlateCarree())
ax.coastlines(color='gray')
ax.set_title('Meridional MSE advection')

ax = fig.add_subplot(gs[1,2],projection=ccrs.PlateCarree())
(-ds.vertical_adv_mse).plot(ax=ax, x='longitude', y='latitude', cmap='RdBu_r',add_colorbar=False,\
                      vmin=-vminmax,vmax=vminmax,transform=ccrs.PlateCarree())
ax.coastlines(color='gray')
ax.set_title('Vertical MSE advection')

ax = fig.add_subplot(gs[2,0],projection=ccrs.PlateCarree())
(-ds.dx_zonal_mse_flux).plot(ax=ax, x='longitude', y='latitude', cmap='RdBu_r',\
                     vmin=-vminmax*100,vmax=vminmax*100,transform=ccrs.PlateCarree(),\
                     add_colorbar=False)
ax.coastlines(color='gray')
ax.set_title('0.01 * d/dx of zonal MSE flux')

ax = fig.add_subplot(gs[2,1],projection=ccrs.PlateCarree())
(-ds.dy_meridional_mse_flux).plot(ax=ax, x='longitude', y='latitude', cmap='RdBu_r',\
                     vmin=-vminmax*100,vmax=vminmax*100,transform=ccrs.PlateCarree(),\
                     add_colorbar=False)
ax.coastlines(color='gray')
ax.set_title('0.01 * d/dy of meridional MSE flux')

ax = fig.add_subplot(gs[2,2],projection=ccrs.PlateCarree())
(-ds.dp_vertical_mse_flux).plot(ax=ax, x='longitude', y='latitude', cmap='RdBu_r',\
                     vmin=-vminmax*10,vmax=vminmax*10,transform=ccrs.PlateCarree(),\
                     add_colorbar=False)
ax.coastlines(color='gray')
ax.set_title('0.1 * d/dp of vertical MSE flux')

ax = fig.add_subplot(gs[3,0],projection=ccrs.PlateCarree())
ds.mse_tendency_convection.plot(ax=ax, x='longitude', y='latitude', cmap='RdBu_r',\
                                vmin=-vminmax,vmax=vminmax,add_colorbar=False,\
                                transform=ccrs.PlateCarree())
ax.coastlines(color='gray')
ax.set_title('Convection and turbulence')

ax = fig.add_subplot(gs[3,1],projection=ccrs.PlateCarree())
ds.mse_tendency_shortwave.plot(ax=ax, x='longitude', y='latitude', cmap='RdBu_r',\
                                vmin=-vminmax,vmax=vminmax,add_colorbar=False,\
                                transform=ccrs.PlateCarree())
ax.coastlines(color='gray')
ax.set_title('Short-wave radiation')

ax = fig.add_subplot(gs[3,2],projection=ccrs.PlateCarree())
ds.mse_tendency_longwave.plot(ax=ax, x='longitude', y='latitude', cmap='RdBu_r',\
                                vmin=-vminmax,vmax=vminmax,add_colorbar=False,\
                                transform=ccrs.PlateCarree())
ax.coastlines(color='gray')
ax.set_title('Long-wave radiation')

ax = fig.add_subplot(gs[4,0],projection=ccrs.PlateCarree())
ra.plot(ax=ax, x='longitude', y='latitude', cmap='RdBu_r',\
                     vmin=-vminmax,vmax=vminmax,transform=ccrs.PlateCarree(),\
                     add_colorbar=False)
ax.coastlines(color='gray')
ax.set_title('$R_a$')

ax = fig.add_subplot(gs[4,1],projection=ccrs.PlateCarree())
sensible_heat_flux.plot(ax=ax, x='longitude', y='latitude', cmap='RdBu_r',\
                     vmin=-vminmax,vmax=vminmax,transform=ccrs.PlateCarree(),\
                     add_colorbar=False)
ax.coastlines(color='gray')
ax.set_title('Sensible heat flux') 

ax = fig.add_subplot(gs[4,2],projection=ccrs.PlateCarree())
latent_heat_flux.plot(ax=ax, x='longitude', y='latitude', cmap='RdBu_r',\
                     vmin=-vminmax,vmax=vminmax,transform=ccrs.PlateCarree(),\
                     add_colorbar=False)
ax.coastlines(color='gray')
ax.set_title('Latent heat flux')





cax = fig.add_subplot(gs[:,-1])
cbar = plt.colorbar(cs, cax=cax,label='$W m^{-2}$')

plt.savefig(plot_path+f'mse_budget_latlon_maps_{plot_timestring}.png', dpi=300, bbox_inches='tight')