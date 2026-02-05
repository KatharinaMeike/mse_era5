# Plot monthly mean advection terms and fluxes
# processed with existing data on 37 pressure levels
#%%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

data_path = '/media/katharina/Volume/UChicago/ERA5/mse_output/monmean/'

plot_path = '/media/katharina/Volume/UChicago/MSE/plots/casestudy/'

plot_timestring = '30days_jan2005'

#%%
# Latitude-level figure

ds = xr.open_dataset(data_path+'mse_daily_P37_varp_levlat_20050102_20050131.nc')

vminmax = 0.05
axes = []

fig = plt.figure(figsize=(13,6))
gs = gridspec.GridSpec(2, 4, width_ratios=[1,1,1,0.03], wspace=0.5,hspace=0.3)
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
vminmax = 1e6
axes = []
fig = plt.figure(figsize=(13,3.5))
gs = gridspec.GridSpec(1, 4, width_ratios=[1,1,1,0.03], wspace=0.5,hspace=0.3)
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
(-10.0 * ds.vertical_mse_flux).plot(ax=ax,cmap='RdBu_r',vmin=-vminmax,vmax=vminmax,\
                        add_colorbar=False)
ax.set_title('10 * $(-m\omega)$ [J kg$^{-1}$ Pa s$^{-1}$]',fontsize=10)

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

#%%
# Longitude-latitude figure

ds = xr.open_dataset(data_path+'mse_daily_P37_varp_latlon_20050102_20050131.nc')

vminmax = 500

fig = plt.figure(figsize=(13,6))
gs = gridspec.GridSpec(2, 4, width_ratios=[1,1,1,0.03], wspace=0.05,hspace=0.2)
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

cax = fig.add_subplot(gs[:,-1])
cbar = plt.colorbar(cs, cax=cax,label='$W m^{-2}$')

plt.savefig(plot_path+f'mse_budget_latlon_maps_{plot_timestring}.png', dpi=300, bbox_inches='tight')




