# Calculate monthly mean temperature on 37 pressure levels
#%%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

data_path = '/project2/tas1/abacus/data1/tas/archive/Reanalysis/ERA5/'

times = pd.date_range('1980-01-01','2022-12-31',freq='MS')

ds_first = xr.open_dataset(data_path+'t/era5_t_1980_01.6hrly.nc')

#%%

# Output dataset
out_dict = {}
out_dict['t'] = (('time','level','latitude','longitude'), np.zeros((len(times), len(ds_first.level), len(ds_first.latitude),len(ds_first.longitude))))

ds_out = xr.Dataset(out_dict, coords={'time':times,
                                      'level':ds_first.level,\
                                     'latitude':ds_first.latitude,\
                                     'longitude':ds_first.longitude})

for i, timestep in enumerate(times):
    ds = xr.open_dataset(data_path+'t/era5_t_'+str(timestep.year)+'_'+str(timestep.month).zfill(2)+'.6hrly.nc')
    ds_out['t'][i] = ds.mean(dim='time').values

ds_out.to_netcdf('/project2/tas1/katharinah/ERA5/t/temperature_monmean_1980_2022.nc')


#%%
"""
Comparison of data for different averaging periods
ds_stf = xr.open_dataset(data_path+'era5_stf_1979_2019.nc')

ds_6h = xr.open_dataset(data_path+'stf/era5_stf_2017_12.6hrly.nc')

monmean = ds_6h.mean(dim='valid_time')

#%%
# Downloaded surface latent heat flux
ds_slhf = xr.open_dataset('/project2/tas1/katharinah/ERA5/slhf/era5_slhf_2017_12_monthlymean.nc')
"""



