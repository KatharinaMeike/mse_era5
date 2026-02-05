#%%
import numpy as np

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

average_hours = -2 # If >0, average over timestep + average_hours hours

data_path = '/media/katharina/Volume/UChicago/ERA5/mse_output/'
plot_path = '/media/katharina/Volume/UChicago/MSE/plots/casestudy/'

# Read MSE advection, tendencies and fluxes
ds_levlat = xr.open_mfdataset(data_path+'with_time_coordinate/*levlat*.nc')
if average_hours > 0:
    ds = ds_levlat.sel(time=slice(timestep,timestep + pd.Timedelta(hours=average_hours))).mean(dim='time')
    plot_timestring = f'{timestep:%Y-%m-%d %H} to {timestep + pd.Timedelta(hours=average_hours):%Y-%m-%d %H} UTC'
elif average_hours == -1: # daily mean
    ds = xr.open_dataset('/media/katharina/Volume/UChicago/ERA5/mse_output/mse_daily_P137_levlat_2005060300.nc')
    plot_timestring = '2005060300_dailymean'
elif average_hours == -2: # daily mean, p37
    ds = xr.open_dataset('/media/katharina/Volume/UChicago/ERA5/mse_output/mse_daily_P37_levlat_2005060300.nc')
    plot_timestring = '2005060300_dailymean_P37'
else:
    ds = ds_levlat.sel(time=timestep)
    plot_timestring = f'{timestep:%Y-%m-%d %H} UTC'

#%%
# Figure of MSE budget

vminmax = 0.05
axes = []

fig = plt.figure(figsize=(13,12))
gs = gridspec.GridSpec(4, 4, width_ratios=[1,1,1,0.03], wspace=0.4,hspace=0.3)
ax = fig.add_subplot(gs[0,0])
axes.append(ax)
cs = ds.mse_tendency.plot(ax=ax,vmin=-vminmax,vmax=vminmax,cmap='RdBu_r',\
                        add_colorbar=False)
ax.set_title('MSE Tendency')

total_advection = -(ds.zonal_adv_mse + ds.meridional_adv_mse + ds.vertical_adv_mse)
ax = fig.add_subplot(gs[0,1])
axes.append(ax)
cs = total_advection.plot(ax=ax,vmin=-vminmax,vmax=vminmax,cmap='RdBu_r',\
                        add_colorbar=False)
ax.set_title('MSE Advection')

total_fluxdiv = -(ds.dx_zonal_mse_flux + ds.dy_meridional_mse_flux + ds.dp_vertical_mse_flux)
ax = fig.add_subplot(gs[0,2])
axes.append(ax)
cs = total_fluxdiv.plot(ax=ax,vmin=-vminmax,vmax=vminmax,cmap='RdBu_r',\
                        add_colorbar=False)
ax.set_title('MSE Flux Divergence')

ax = fig.add_subplot(gs[1,0])
axes.append(ax)
(-ds.zonal_adv_mse).plot(ax=ax,vmin=-vminmax,vmax=vminmax,cmap='RdBu_r',\
                        add_colorbar=False)
ax.set_title('Zonal MSE Advection')

ax = fig.add_subplot(gs[1,1])
axes.append(ax)
(-ds.meridional_adv_mse).plot(ax=ax,vmin=-vminmax,vmax=vminmax,cmap='RdBu_r',\
                        add_colorbar=False)
ax.set_title('Meridional MSE Advection')

ax = fig.add_subplot(gs[1,2])
axes.append(ax)
(-ds.vertical_adv_mse).plot(ax=ax,vmin=-vminmax,vmax=vminmax,cmap='RdBu_r',\
                        add_colorbar=False)
ax.set_title('Vertical MSE Advection')

ax = fig.add_subplot(gs[2,0])
axes.append(ax)
(-ds.dx_zonal_mse_flux).plot(ax=ax,vmin=-vminmax*10,vmax=vminmax*10,cmap='RdBu_r',\
                        add_colorbar=False)
ax.set_title('0.1 * Zonal MSE Flux Divergence')

ax = fig.add_subplot(gs[2,1])
axes.append(ax)
(-ds.dy_meridional_mse_flux).plot(ax=ax,vmin=-vminmax*10,vmax=vminmax*10,cmap='RdBu_r',\
                        add_colorbar=False)
ax.set_title('0.1 * Meridional MSE Flux Divergence')

ax = fig.add_subplot(gs[2,2])
axes.append(ax)
(-ds.dp_vertical_mse_flux).plot(ax=ax,vmin=-vminmax*10,vmax=vminmax*10,cmap='RdBu_r',\
                        add_colorbar=False)
ax.set_title('0.1 * Vertical MSE Flux Divergence')

ax = fig.add_subplot(gs[3,0])
axes.append(ax)
ds.mse_tendency_convection.plot(ax=ax,vmin=-vminmax,vmax=vminmax,cmap='RdBu_r',\
                        add_colorbar=False)
ax.set_title('Convection and turbulence')

ax = fig.add_subplot(gs[3,1])
axes.append(ax)
ds.mse_tendency_shortwave.plot(ax=ax,vmin=-vminmax,vmax=vminmax,cmap='RdBu_r',\
                        add_colorbar=False)
ax.set_title('Shortwave radiation')

ax = fig.add_subplot(gs[3,2])
axes.append(ax)
ds.mse_tendency_longwave.plot(ax=ax,vmin=-vminmax,vmax=vminmax,cmap='RdBu_r',\
                        add_colorbar=False)
ax.set_title('Longwave radiation')



cax = fig.add_subplot(gs[:,-1])
cbar = plt.colorbar(cs, cax=cax,label='W kg$^{-1}$')

for ax in axes:
    ax.invert_yaxis()
    ax.set_yscale('log')
    ax.set_ylim(1000,0.01)
    ax.set_ylabel('Pressure [hPa]')
    ax.set_xlabel('')
    ax.set_xticks([ -60, -30, 0, 30, 60])
    ax.set_xticklabels([ '60°S', '30°S', 'EQ', '30°N', '60°N'])

plt.savefig(plot_path+f'mse_budget_levlat_{plot_timestring}_full.png', dpi=300, bbox_inches='tight')

# Save a second version with troposphere focus (100-1000 hPa)
for ax in axes:
    ax.set_ylim(1000,100)

plt.savefig(plot_path+f'mse_budget_levlat_{plot_timestring}_troposphere.png', dpi=300, bbox_inches='tight')

#%%
# Figure of MSE fluxes
vminmax = 2e6
axes = []
fig = plt.figure(figsize=(13,6))
gs = gridspec.GridSpec(2, 4, width_ratios=[1,1,1,0.03], wspace=0.5,hspace=0.3)
ax = fig.add_subplot(gs[0,0])
axes.append(ax)
cs = (-0.05 * ds.zonal_mse_flux).plot(ax=ax,cmap='RdBu_r',vmin=-vminmax,vmax=vminmax,\
                        add_colorbar=False)
ax.set_title('0.05 * $(-mu)$ [J kg$^{-1}$ m s$^{-1}$]',fontsize=10)

ax = fig.add_subplot(gs[0,1])
axes.append(ax)
cs = (-1.0 * ds.meridional_mse_flux).plot(ax=ax,cmap='RdBu_r',vmin=-vminmax,vmax=vminmax,\
                        add_colorbar=False)
ax.set_title('$-mv$ [J kg$^{-1}$ m s$^{-1}$]',fontsize=10)

ax = fig.add_subplot(gs[0,2])
axes.append(ax)
(-100.0 * ds.vertical_mse_flux).plot(ax=ax,cmap='RdBu_r',vmin=-vminmax,vmax=vminmax,\
                        add_colorbar=False)
ax.set_title('100 * $(-m\omega)$ [J kg$^{-1}$ Pa s$^{-1}$]',fontsize=10)

ax = fig.add_subplot(gs[1,0])
axes.append(ax)
(1000.0 * ds.convection_mse_flux).plot(ax=ax, vmin=-vminmax, vmax=vminmax, cmap='RdBu_r',\
                        add_colorbar=False)
ax.set_title('1000 * Convective MSE flux [J kg$^{-1}$ Pa s$^{-1}$]',fontsize=10)

ax = fig.add_subplot(gs[1,1])
axes.append(ax)
(1000.0 * ds.shortwave_mse_flux).plot(ax=ax, vmin=-vminmax, vmax=vminmax, cmap='RdBu_r',\
                        add_colorbar=False)
ax.set_title('1000 * Short-wave MSE flux [J kg$^{-1}$ Pa s$^{-1}$]',fontsize=10)

ax = fig.add_subplot(gs[1,2])
axes.append(ax)
(1000.0 * ds.longwave_mse_flux).plot(ax=ax, vmin=-vminmax, vmax=vminmax, cmap='RdBu_r',\
                        add_colorbar=False)
ax.set_title('1000 * Long-wave MSE flux [J kg$^{-1}$ Pa s$^{-1}$]',fontsize=10)

cax = fig.add_subplot(gs[:,-1])
cbar = plt.colorbar(cs, cax=cax)


for ax in axes:
    ax.invert_yaxis()
    ax.set_yscale('log')
    ax.set_ylim(1000,0.01)
    ax.set_ylabel('Pressure [hPa]')
    ax.set_xlabel('')
    ax.set_xticks([ -60, -30, 0, 30, 60])
    ax.set_xticklabels([ '60°S', '30°S', 'EQ', '30°N', '60°N'])

plt.savefig(plot_path+f'mse_fluxes_levlat_{plot_timestring}_full.png', dpi=300, bbox_inches='tight')

# Save a second version with troposphere focus (100-1000 hPa)
for ax in axes:
    ax.set_ylim(1000,100)

plt.savefig(plot_path+f'mse_fluxes_levlat_{plot_timestring}_troposphere.png', dpi=300, bbox_inches='tight')

