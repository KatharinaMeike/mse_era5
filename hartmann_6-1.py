# For which time intervals do I get Fig. 6.1 in Hartmann (2016)?
#%%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

data_path = '/media/katharina/Volume/UChicago/ERA5/'

plot_path = '/media/katharina/Volume/UChicago/MSE/plots/long_time/'

sec_per_day = 86400.0

# Surface heat fluxes
ds_stf = xr.open_dataset(data_path+'era5_stf_1979_2019.nc')

# Radiative quantities
ds_rad = xr.open_dataset(data_path+'era5_rad_1979_2019.nc')

#%%

start_time = '2005-06-01'
end_time = '2005-06-30'

latent_heat_flux = -ds_stf.slhf.sel(time=slice(start_time,end_time)).mean(dim=['time','longitude']) / sec_per_day
sensible_heat_flux = -ds_stf.sshf.sel(time=slice(start_time,end_time)).mean(dim=['time','longitude']) / sec_per_day

ra = (ds_rad.tsr + ds_rad.ttr - ds_rad.ssr - ds_rad.str).sel(time=slice(start_time,end_time)).mean(dim=['time','longitude']) / sec_per_day 

fig, ax = plt.subplots(figsize=(6,6))
latent_heat_flux.plot(ax=ax,y='latitude',label='Latent Heat Flux',color='blue')
sensible_heat_flux.plot(ax=ax,y='latitude',label='Sensible Heat Flux',color='orange')
ra.plot(ax=ax,y='latitude',label='Net Radiative Flux',color='k')
(latent_heat_flux + sensible_heat_flux + ra).plot(ax=ax,y='latitude',label='Advection (residual)',color='red',linestyle='dashed')
ax.axvline(0,color='gray',linestyle='dotted')
ax.set_ylabel('W m$^{-2}$')
plt.savefig(plot_path+'hartmann_6-1_fluxes_'+start_time+'-'+end_time+'.pdf',bbox_inches='tight')


#%%
# Only June 2005



#%%

ra = ds_rad.tsr + ds_rad.ttr - ds_rad.ssr - ds_rad.str

# Zonal averages
ra_zonalmean = ra.mean(dim='longitude')
sh_zonalmean = ds_stf.sshf.mean(dim='longitude')
lh_zonalmean = ds_stf.slhf.mean(dim='longitude')

# Compute R1
r1 = 1.0 - (sh_zonalmean+ lh_zonalmean) / ra_zonalmean




