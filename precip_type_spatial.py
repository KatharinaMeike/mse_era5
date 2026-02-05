# Compare convective and large-scale precipitation spatial patterns
#%%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs


data_path = '/media/katharina/Volume/UChicago/ERA5/precip_type/'

ds = xr.open_mfdataset(data_path+'era5_mlspr_mcpr_*.nc')

# Time average of precipitation types
convective_mean = ds.avg_cpr.mean(dim='valid_time')
largescale_mean = ds.avg_lsprate.mean(dim='valid_time')

#%%

# Plot spatial patterns
fig = plt.figure(figsize=(7,10))
gs = gridspec.GridSpec(3, 1, hspace=0.3)
ax = fig.add_subplot(gs[0],projection=ccrs.PlateCarree())
cs = convective_mean.plot(ax=ax, x='longitude', y='latitude', cmap='Blues',\
                         vmin=0,vmax=1e-4,transform=ccrs.PlateCarree(),\
                         cbar_kwargs={'label':'mm s$^{-1}$'})
ax.coastlines(color='gray')
ax.set_title('Convective Precipitation Rate')

ax = fig.add_subplot(gs[1],projection=ccrs.PlateCarree())
cs = largescale_mean.plot(ax=ax, x='longitude', y='latitude', cmap='Blues',\
                         vmin=0,vmax=1e-4,transform=ccrs.PlateCarree(),\
                         cbar_kwargs={'label':'mm s$^{-1}$'})
ax.coastlines(color='gray')
ax.set_title('Large-scale Precipitation Rate')


ax = fig.add_subplot(gs[2],projection=ccrs.PlateCarree())
total_precip = convective_mean + largescale_mean
cs = (convective_mean / total_precip).plot(ax=ax, x='longitude', y='latitude', cmap='viridis',\
                         vmin=0,vmax=1,transform=ccrs.PlateCarree(),\
                         levels=np.arange(0, 1.1, 0.1),\
                         cbar_kwargs={'label':'Fraction of Convective Precipitation'})
ax.coastlines(color='gray')
ax.set_title('Fraction of Convective Precipitation')

plt.savefig('/media/katharina/Volume/UChicago/MSE/plots/precip/convective_fraction_spatial.png',dpi=300,bbox_inches='tight')

#%%
# Monthly mean values
convective_monthly = ds.avg_cpr.groupby('valid_time.month').mean(dim='valid_time')
largescale_monthly = ds.avg_lsprate.groupby('valid_time.month').mean(dim='valid_time')

fig = plt.figure(figsize=(15,30))
gs = gridspec.GridSpec(12, 3, hspace=0.4, wspace=0.3)
for month in range(1,13):
    ax = fig.add_subplot(gs[month-1,0],projection=ccrs.PlateCarree())
    cs = convective_monthly.sel(month=month).plot(ax=ax, x='longitude', y='latitude', cmap='Blues',\
                             vmin=0,vmax=1e-4,transform=ccrs.PlateCarree(),\
                             cbar_kwargs={'label':'mm s$^{-1}$'})
    ax.coastlines(color='gray')
    ax.set_title(f'Convective Precipitation Rate - Month {month}')

    ax = fig.add_subplot(gs[month-1,1],projection=ccrs.PlateCarree())
    cs = largescale_monthly.sel(month=month).plot(ax=ax, x='longitude', y='latitude', cmap='Blues',\
                             vmin=0,vmax=1e-4,transform=ccrs.PlateCarree(),\
                             cbar_kwargs={'label':'mm s$^{-1}$'})
    ax.coastlines(color='gray')
    ax.set_title(f'Large-scale Precipitation Rate - Month {month}')

    ax = fig.add_subplot(gs[month-1,2],projection=ccrs.PlateCarree())
    total_precip_month = convective_monthly.sel(month=month) + largescale_monthly.sel(month=month)
    cs = (convective_monthly.sel(month=month) / total_precip_month).plot(ax=ax, x='longitude', y='latitude', cmap='viridis',\
                             vmin=0,vmax=1,transform=ccrs.PlateCarree(),\
                             levels=np.arange(0, 1.1, 0.1),\
                             cbar_kwargs={'label':'Fraction of Convective Precipitation'})
    ax.coastlines(color='gray')
    ax.set_title(f'Fraction of Convective Precipitation - Month {month}')


#%%
# Montly means of convective precipitation fraction
months = ['January','February','March','April','May','June',\
          'July','August','September','October','November','December']
fig, axes = plt.subplots(4,3,figsize=(15,10),subplot_kw={'projection': ccrs.PlateCarree()})
axes = axes.flatten()
for month in range(1,13):
    ax = axes[month-1]
    total_precip_month = convective_monthly.sel(month=month) + largescale_monthly.sel(month=month)
    cs = (convective_monthly.sel(month=month) / total_precip_month).plot(ax=ax, x='longitude', y='latitude', cmap='viridis',\
                             vmin=0,vmax=1,transform=ccrs.PlateCarree(),\
                             levels=np.arange(0, 1.1, 0.1),\
                             add_colorbar=False)
    ax.coastlines(color='gray')
    ax.set_title(f'{months[month-1]}')
plt.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(cs, cax=cbar_ax, label='Fraction of Convective Precipitation')
plt.savefig('/media/katharina/Volume/UChicago/MSE/plots/precip/convective_fraction_monthly.png',dpi=300,bbox_inches='tight')


