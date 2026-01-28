# Convective and large-scale precipitation rate
#%%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

plot_path = '/project2/tas1/katharinah/mse/precip/plots/'
data_path = '/project2/tas1/abacus/data1/tas/archive/Reanalysis/ERA5/'

#%%

# Total precipitation
ds_mtpr = xr.open_dataset(data_path+'mtpr/era5_mtpr_1979_2020_monthlymean.nc').sel(time=slice('1980-01-01','2005-12-31'))

# Convective and large-scale precipitation
precip_path = '/project2/tas1/katharinah/ERA5/precip_type/'
pr_1980 = xr.open_dataset(precip_path+'era5_mlspr_mcpr_1980_2000_monthlymean.nc')
pr_2001 = xr.open_dataset(precip_path+'era5_mlspr_mcpr_2001_2005_monthlymean.nc')
ds_pr = xr.concat([pr_1980,pr_2001],dim='valid_time').drop_vars(['number','expver']).rename({'valid_time':'time'}).assign_coords(time=ds_mtpr.time)

#%%
# Monthly climatology of zonally averaged precipitation
conv_clim = ds_pr.avg_cpr.mean(dim='longitude').groupby('time.month').mean(dim='time')
ls_clim = ds_pr.avg_lsprate.mean(dim='longitude').groupby('time.month').mean(dim='time')

#%%
# Plot precipitation climatology
fig, ax = plt.subplots(figsize=(5,3))
conv_clim.plot(ax=ax,y='latitude',cmap='Blues',cbar_kwargs={'label':'Convective precipitation rate [kg m$^{-2}$ s$^{-1}$]'},vmin=0,vmax=3e-5)
ax.set_xlabel('Month')
ax.set_ylabel('Latitude [degrees]')
plt.savefig(plot_path+'conv_clim.png',dpi=300,bbox_inches='tight')

fig, ax = plt.subplots(figsize=(5,3))
ls_clim.plot(ax=ax,y='latitude',cmap='Blues',cbar_kwargs={'label':'Large-scale precipitation rate [kg m$^{-2}$ s$^{-1}$]'},vmin=0,vmax=3e-5)
ax.set_xlabel('Month')
ax.set_ylabel('Latitude [degrees]')
plt.savefig(plot_path+'largescale_clim.png',dpi=300,bbox_inches='tight')

#%%
# Fraction of convective and large-scale precipitation computed using zonal averages
convective_fraction = ds_pr.avg_cpr.mean(dim='longitude') / ds_mtpr.mtpr.mean(dim='longitude')
largescale_fraction = ds_pr.avg_lsprate.mean(dim='longitude') / ds_mtpr.mtpr.mean(dim='longitude')

conv_frac_clim = convective_fraction.groupby('time.month').mean(dim='time')
ls_frac_clim = largescale_fraction.groupby('time.month').mean(dim='time')

#%%
fig, ax = plt.subplots(figsize=(5,3))
conv_frac_clim.plot(ax=ax,y='latitude',cmap='Blues',cbar_kwargs={'label':'Convective precipitation fraction [~]'},vmin=0,vmax=1)
ax.set_xlabel('Month')
ax.set_ylabel('Latitude [degrees]')
plt.savefig(plot_path+'conv_frac_clim.png',dpi=300,bbox_inches='tight')

#%%
fig, ax = plt.subplots(figsize=(5,3))
ls_frac_clim.plot(ax=ax,y='latitude',cmap='Blues',cbar_kwargs={'label':'Large-scale precipitation fraction [~]'},vmin=0,vmax=1)
ax.set_xlabel('Month')
ax.set_ylabel('Latitude [degrees]')
plt.savefig(plot_path+'ls_frac_clim.png',dpi=300,bbox_inches='tight')

#%%
# Snow etc. (I guess this is not really snow, probably snow is part of the large-scale and convective precipitation)
fig, ax = plt.subplots(figsize=(5,3))
(-ls_frac_clim - conv_frac_clim + 1.0).plot(ax=ax,y='latitude',cmap='Blues',cbar_kwargs={'label':'Snow etc. fraction [~]'},vmin=0,vmax=1)
ax.set_xlabel('Month')
ax.set_ylabel('Latitude [degrees]')
plt.savefig(plot_path+'snow_frac_clim.png',dpi=300,bbox_inches='tight')

#%%
# Fraction of convective precipitation - this can be larger than 1 due to inaccuracies in regions of small values, be careful!
convective_fraction = ds_pr.avg_cpr / ds_mtpr.mtpr


# ---------------------------------------------------------
#%%
# Comparison of seasonality of convective precipitation rate and R1
# Surface heat fluxes
ds_stf = xr.open_dataset(data_path+'era5_stf_1979_2019.nc')

# Radiative quantities
ds_rad = xr.open_dataset(data_path+'era5_rad_1979_2019.nc')

ra = ds_rad.tsr + ds_rad.ttr - ds_rad.ssr - ds_rad.str

# Zonal averages
ra_zonalmean = ra.mean(dim='longitude')
sh_zonalmean = ds_stf.sshf.mean(dim='longitude')
lh_zonalmean = ds_stf.slhf.mean(dim='longitude')

# Compute R1
r1 = 1.0 - (sh_zonalmean+ lh_zonalmean) / ra_zonalmean
r1_monthly_avg = r1.sel(time=slice('1980-01-01','2005-12-31')).groupby('time.month').mean(dim='time')

#%%

def largescale_conv_r1(lat_north=60,lat_south=40,plot_rce=True, plot_rae=True,ymin=-0.6,ymax=1.2,ymin_conv=0.4,ymax_conv=0.8):
    fig, ax = plt.subplots(figsize=(5,4))
    r1_monthly_avg.sel(latitude=slice(lat_north,lat_south)).mean(dim='latitude').plot(ax=ax,color='k')
    ax2 = ax.twinx()
    ls_frac_clim.sel(latitude=slice(lat_north,lat_south)).mean(dim='latitude').plot(ax=ax2,color='blue')
    if plot_rae:
        ax.fill_between(np.linspace(1,12,100),0.9,ymax,color='lightblue',alpha=0.5)
    if plot_rce:
        ax.fill_between(np.linspace(1,12,100),ymin,0.1,color='darkorange',alpha=0.5)
    ax.set_xlim(1,12)
    ax.set_ylim(ymin,ymax)
    ax2.set_ylim(ymin_conv,ymax_conv)
    ax.set_ylabel('$R_1$')
    ax2.set_ylabel('Large-scale precipitation fraction',color='blue')
    ax2.set_yticks(np.arange(ymin_conv,ymax_conv+0.1,0.1))
    ax2.set_yticklabels(np.round(np.arange(ymin_conv,ymax_conv+0.1,0.1),2),color='blue')

    plt.savefig(plot_path+'lat_'+str(lat_north)+'_'+str(lat_south)+'.pdf',bbox_inches='tight')

largescale_conv_r1()

#%%
largescale_conv_r1(lat_north=70,lat_south=60,ymin=-0.1,ymax=1.5,ymin_conv=0.5,ymax_conv=1.0)
largescale_conv_r1(lat_north=90,lat_south=70,ymin=0.6,ymax=1.1,ymin_conv=0.8,ymax_conv=1.0)
largescale_conv_r1(lat_north=40,lat_south=30,ymax=0.2,ymin=-0.5,ymax_conv=0.6,ymin_conv=0.3)
largescale_conv_r1(lat_north=30,lat_south=10,ymax=0.2,ymin=-0.5,ymax_conv=0.6,ymin_conv=0.3)
largescale_conv_r1(lat_north=10,lat_south=-10,ymax=0.2,ymin=-0.5,ymax_conv=0.4,ymin_conv=0.2)
largescale_conv_r1(lat_north=-10,lat_south=-30,ymax=0.2,ymin=-0.5,ymax_conv=0.6,ymin_conv=0.3)
largescale_conv_r1(lat_north=-30,lat_south=-40,ymax=0.2,ymin=-0.5,ymax_conv=0.6,ymin_conv=0.3)
largescale_conv_r1(lat_north=-40,lat_south=-60,ymax=0.8,ymin=0.2,ymax_conv=0.8,ymin_conv=0.5)
largescale_conv_r1(lat_north=-60,lat_south=-70,ymax=1.0,ymin=0.6,ymax_conv=1.0,ymin_conv=0.8)
largescale_conv_r1(lat_north=-70,lat_south=-90,ymax=1.4,ymin=1.0,ymax_conv=1.0,ymin_conv=0.9)


#%%
# Large-scale precipitation fraction as a function of latitude

ymin_conv = 0.2
ymax_conv = 1.1

r1_tavg = r1.sel(time=slice('1980-01-01','2005-12-31')).mean(dim='time')
conv_tavg = convective_fraction.mean(dim='time')
ls_tavg = largescale_fraction.mean(dim='time')

fig, ax = plt.subplots(figsize=(5,3))
r1_tavg.plot(ax=ax,color='k')
ax2 = ax.twinx()
ax2.set_ylim(ymin_conv,ymax_conv)
ax2.set_ylabel('Large-scale precipitation fraction',color='blue')
ax2.set_yticks(np.arange(ymin_conv,ymax_conv+0.1,0.1))
ax2.set_yticklabels(np.round(np.arange(ymin_conv,ymax_conv+0.1,0.1),2),color='blue')
ls_tavg.plot(ax=ax2,color='blue')
ax.set_ylabel('$R_1$ [~]')
ax.set_xlabel('Latitude [degrees]')
ax.fill_between(np.linspace(-90,90,100),0.9,1.8,color='lightblue',alpha=0.5)
ax.fill_between(np.linspace(-90,90,100),-0.6,0.1,color='darkorange',alpha=0.5)
ax.set_ylim(-0.6,1.8)
ax.set_xlim(-90,90)
plt.savefig(plot_path+'annual_average_precip_r1.pdf',dpi=300,bbox_inches='tight')

#%%

