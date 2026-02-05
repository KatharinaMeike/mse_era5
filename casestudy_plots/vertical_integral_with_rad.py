# For 2 days (case study): Plot latitude-dependent vertically integrated quantities
# to compare with top-of-atmosphere and surface fluxes
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

timestep = pd.Timestamp('2005-06-02 07:00')

average_hours = 23 # If >0, average over timestep + average_hours hours

data_path = '/media/katharina/Volume/UChicago/ERA5/mse_output/'
plot_path = '/media/katharina/Volume/UChicago/MSE/plots/casestudy/'

# Read MSE advection, tendencies and fluxes
ds_levlat = xr.open_mfdataset(data_path+'with_time_coordinate/*latlon*.nc')
if average_hours > 0:
    ds = ds_levlat.sel(time=slice(timestep,timestep + pd.Timedelta(hours=average_hours))).mean(dim='time')
    plot_timestring = f'{timestep:%Y-%m-%d %H} to {timestep + pd.Timedelta(hours=average_hours):%Y-%m-%d %H} UTC'
else:
    ds = ds_levlat.sel(time=timestep)
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

# R1 as in Miyawaki et al. (2022)
r1 = 1.0 + (sensible_heat_flux.mean(dim='longitude') + latent_heat_flux.mean(dim='longitude')) \
    / ra.mean(dim='longitude')

#%%
# Advection of MSE
mse_advection = (ds.zonal_adv_mse.mean(dim='longitude') +\
                 ds.meridional_adv_mse.mean(dim='longitude') +\
                 ds.vertical_adv_mse.mean(dim='longitude'))

tendency = ds.mse_tendency.mean(dim='longitude')

residual = -tendency - mse_advection + sensible_heat_flux.mean(dim='longitude') +\
      latent_heat_flux.mean(dim='longitude') + ra.mean(dim='longitude')

fig, ax = plt.subplots(figsize=(6,4),sharey=True)
sensible_heat_flux.mean(dim='longitude').plot(ax=ax, y='latitude', label='SH', color='C2')
latent_heat_flux.mean(dim='longitude').plot(ax=ax, y='latitude', label='LH', color='blue')
(tendency +mse_advection).plot(ax=ax, y='latitude', label='$\partial_t m + \partial_y (vm)$', color='red')
ra.mean(dim='longitude').plot(ax=ax, y='latitude', label='$R_a$', color='C4')
residual.plot(ax=ax, y='latitude', label='$-\partial_t m - \partial_y (vm)$ + SH + LH + $R_a$', color='k', linestyle='--')
mse_advection.plot(ax=ax, y='latitude', label='$\partial_y (vm)$', color='C0',linewidth=0.5)
tendency.plot(ax=ax, y='latitude', label='$\partial_t m$', color='C1',linewidth=0.5)
ax.axvline(0, color='gray')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_title('')
ax.set_xlabel('MSE tendency [W m$^{-2}$]')
ax.set_ylabel('')
ax.set_xlim(-300,300)
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
ax.set_yticklabels(['90°S', '60°S', '30°S', '0', '30°N', '60°N', '90°N'])
#ax = axes[1]
#r1.plot(ax=ax, y='latitude', label='R1', color='k')
#ax.set_ylabel('')
#ax.set_xlabel('R1 [~]')
#ax.axvline(0.1,color='orange')
#ax.axvline(0.9,color='lightblue')
#ax.set_title('')
plt.savefig(plot_path+f'vertical_integral_{plot_timestring.replace(" ", "_")}.png', bbox_inches='tight', dpi=300)

#%%

# Same plot as above, but split into short-wave and long-wave radiation
fig, axes = plt.subplots(ncols=2, figsize=(8,4),sharey=True,width_ratios=[3,1])
ax = axes[0]
mse_advection.plot(ax=ax, y='latitude', label='MSE Advection', color='C0')
tendency.plot(ax=ax, y='latitude', label='MSE Tendency', color='C1')
sensible_heat_flux.mean(dim='longitude').plot(ax=ax, y='latitude', label='Sensible Heat Flux', color='C2')
latent_heat_flux.mean(dim='longitude').plot(ax=ax, y='latitude', label='Latent Heat Flux', color='C3')
shortwave_single.mean(dim='longitude').plot(ax=ax, y='latitude', label='Short-wave Radiation', color='C5')
longwave_single.mean(dim='longitude').plot(ax=ax, y='latitude', label='Long-wave Radiation', color='C6')
residual.plot(ax=ax, y='latitude', label='Residual', color='k', linestyle='--')
ax.axvline(0, color='gray')
ax.legend(loc='upper right',fontsize='small')
ax.set_title('')
ax.set_xlabel('MSE tendency [W m$^{-2}$]')
ax.set_ylabel('')
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
ax.set_yticklabels(['90°S', '60°S', '30°S', '0', '30°N', '60°N', '90°N'])
ax.set_xlim(-300,500)
ax = axes[1]
r1.plot(ax=ax, y='latitude', label='R1', color='k')
ax.set_ylabel('')
ax.set_xlabel('R1 [~]')
ax.axvline(0.1,color='orange')
ax.axvline(0.9,color='lightblue')
ax.set_title('')
plt.savefig(plot_path+f'vertical_integral_shortlong_{plot_timestring.replace(" ", "_")}.png', bbox_inches='tight', dpi=300)

#%%

# MSE budget in flux form
mse_fluxdiv = -(ds.dx_zonal_mse_flux.mean(dim='longitude') +\
               ds.dy_meridional_mse_flux.mean(dim='longitude') +\
               ds.dp_vertical_mse_flux.mean(dim='longitude'))

tendency = ds.mse_tendency.mean(dim='longitude')

residual = mse_fluxdiv - tendency + sensible_heat_flux.mean(dim='longitude') +\
      latent_heat_flux.mean(dim='longitude') + ra.mean(dim='longitude')

fig, axes = plt.subplots(ncols=2, figsize=(8,4),sharey=True,width_ratios=[3,1])
ax = axes[0]
mse_fluxdiv.plot(ax=ax, y='latitude', label='MSE Flux Divergence', color='C0',alpha=0.5)
tendency.plot(ax=ax, y='latitude', label='MSE Tendency', color='C1')
sensible_heat_flux.mean(dim='longitude').plot(ax=ax, y='latitude', label='Sensible Heat Flux', color='C2')
latent_heat_flux.mean(dim='longitude').plot(ax=ax, y='latitude', label='Latent Heat Flux', color='C3')
ra.mean(dim='longitude').plot(ax=ax, y='latitude', label='Net Radiative Flux', color='C4')
residual.plot(ax=ax, y='latitude', label='Residual', color='k', linestyle='--',alpha=0.5)
ax.axvline(0, color='gray')
ax.legend(loc='upper right')
ax.set_title('')
ax.set_xlabel('MSE tendency [W m$^{-2}$]')
ax.set_ylabel('')
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
ax.set_yticklabels(['90°S', '60°S', '30°S', '0', '30°N', '60°N', '90°N'])
ax.set_xlim(-300,500)
ax = axes[1]
r1.plot(ax=ax, y='latitude', label='R1', color='k')
ax.set_ylabel('')
ax.set_xlabel('R1 [~]')
ax.axvline(0.1,color='orange')
ax.axvline(0.9,color='lightblue')
ax.set_title('')
plt.savefig(plot_path+f'vertical_integral_fluxdiv_{plot_timestring.replace(" ", "_")}.png', bbox_inches='tight', dpi=300)

#%%

# Vertically integrated MSE budget with tendencies due to parametrizations
# Advection of MSE
mse_advection = (ds.zonal_adv_mse.mean(dim='longitude') +\
                 ds.meridional_adv_mse.mean(dim='longitude') +\
                 ds.vertical_adv_mse.mean(dim='longitude'))

tendency = ds.mse_tendency.mean(dim='longitude')

residual = -mse_advection - tendency + ds.mse_tendency_longwave.mean(dim='longitude') +\
      ds.mse_tendency_shortwave.mean(dim='longitude') + ds.mse_tendency_convection.mean(dim='longitude')

fig, ax = plt.subplots(figsize=(8,4))
#ax = axes[0]

#ds.mse_tendency_shortwave.mean(dim='longitude').plot(ax=ax, y='latitude', label='Short-wave radiation', color='C5')
#ds.mse_tendency_longwave.mean(dim='longitude').plot(ax=ax, y='latitude', label='Long-wave radiation', color='C6')
(ds.mse_tendency_shortwave.mean(dim='longitude') + ds.mse_tendency_longwave.mean(dim='longitude')).plot(ax=ax, y='latitude', label='$R_a$', color='purple')
(tendency +mse_advection).plot(ax=ax, y='latitude', label='$\partial_t m + \partial_y (vm)$', color='red')
ds.mse_tendency_convection.mean(dim='longitude').plot(ax=ax, y='latitude', label='$c_p P_\mathrm{nR} + L P_q$', color='C7')
residual.plot(ax=ax, y='latitude', label='$-\partial_t m$ - Adv. + $c_p P_\mathrm{nR} + L P_q$ + $R_a$', color='k', linestyle='--')
mse_advection.plot(ax=ax, y='latitude', label='Adv. = $u \partial_x m + v \partial_y m + \omega \partial_p m$', color='C0',linewidth=0.5)
tendency.plot(ax=ax, y='latitude', label='$\partial_t m$', color='C1',linewidth=0.5)
ax.axvline(0, color='gray')
ax.legend(loc='upper right',fontsize='small')
ax.set_title('')
ax.set_xlabel('MSE tendency [W m$^{-2}$]')
ax.set_ylabel('')
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
ax.set_xlim(-400,500)
ax.set_yticklabels(['90°S', '60°S', '30°S', '0', '30°N', '60°N', '90°N'])
"""ax = axes[1]  
r1.plot(ax=ax, y='latitude', label='R1', color='k')
ax.set_ylabel('')
ax.set_xlabel('R1 [~]')
ax.axvline(0.1,color='orange')
ax.axvline(0.9,color='lightblue')
ax.set_title('')"""
plt.savefig(plot_path+f'vertical_integral_convrad_{plot_timestring.replace(" ", "_")}.png', bbox_inches='tight', dpi=300)

#%%

# Components of MSE advection
mse_advection = -(ds.zonal_adv_mse.mean(dim='longitude') +\
                 ds.meridional_adv_mse.mean(dim='longitude') +\
                 ds.vertical_adv_mse.mean(dim='longitude'))

radiative_convection = ds.mse_tendency_shortwave.mean(dim='longitude') +\
        ds.mse_tendency_longwave.mean(dim='longitude') + ds.mse_tendency_convection.mean(dim='longitude')

tendency = ds.mse_tendency.mean(dim='longitude')

residual = mse_advection - tendency + radiative_convection

fig, axes = plt.subplots(ncols=2, figsize=(8,4),sharey=True,width_ratios=[3,1])
ax = axes[0]
(-ds.zonal_adv_mse.mean(dim='longitude')).plot(ax=ax, y='latitude', label='Zonal MSE advection', color='C0')
(-ds.meridional_adv_mse.mean(dim='longitude')).plot(ax=ax, y='latitude', label='Meridional MSE advection', color='C2')
(-ds.vertical_adv_mse.mean(dim='longitude')).plot(ax=ax, y='latitude', label='Vertical MSE advection', color='C3')
tendency.plot(ax=ax, y='latitude', label='MSE Tendency', color='C1')
radiative_convection.plot(ax=ax, y='latitude', label='Radiation and convection', color='C5')
residual.plot(ax=ax, y='latitude', label='Residual', color='k', linestyle='--')
ax.axvline(0, color='gray')
ax.legend(loc='upper right')
ax.set_title('')
ax.set_xlabel('MSE tendency [W m$^{-2}$]')
ax.set_ylabel('')
ax.set_xlim(-500,1000)
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
ax.set_yticklabels(['90°S', '60°S', '30°S', '0', '30°N', '60°N', '90°N'])
ax = axes[1]
r1.plot(ax=ax, y='latitude', label='R1', color='k')
ax.set_ylabel('')
ax.set_xlabel('R1 [~]')
ax.axvline(0.1,color='orange')
ax.axvline(0.9,color='lightblue')
ax.set_title('')
plt.savefig(plot_path+f'vertical_integral_advection_components_{plot_timestring.replace(" ", "_")}.png', bbox_inches='tight', dpi=300)



