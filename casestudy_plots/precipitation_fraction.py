# Plot zonally averaged convective and large-scale precipitation
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

if average_hours > 0:
    plot_timestring = f'{timestep:%Y-%m-%d %H} to {timestep + pd.Timedelta(hours=average_hours):%Y-%m-%d %H} UTC'
else:
    plot_timestring = f'{timestep:%Y-%m-%d %H} UTC'

# Read single-level data (precipitation rates included)
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



convective_precip = ds_tb.avg_cpr.mean(dim='longitude') * 86400.0 # convert to mm/day
large_scale_precip = ds_tb.avg_lsprate.mean(dim='longitude') * 86400.0 # convert to mm/day
total_precip = convective_precip + large_scale_precip
precip_fraction_convective = convective_precip / total_precip
precip_fraction_large_scale = large_scale_precip / total_precip


# Plot precipitation rates
fig, ax = plt.subplots(figsize=(5,4))
convective_precip.plot(ax=ax, y='latitude', label='Convective precipitation rate')
large_scale_precip.plot(ax=ax, y='latitude', label='Large-scale precipitation rate')
ax.legend()
ax.set_title('')
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
ax.set_yticklabels(['90°S', '60°S', '30°S', '0', '30°N', '60°N', '90°N'])
ax.set_ylabel('')
ax.set_xlabel('Precipitation rate (mm/day)')
plt.savefig(plot_path + f'precipitation_rates_{plot_timestring}.png', dpi=300, bbox_inches='tight')


# Plot convective precipitation fraction
fig, ax = plt.subplots(figsize=(5,4))
precip_fraction_convective.plot(ax=ax, y='latitude', label='Convective precipitation fraction')
#precip_fraction_large_scale.plot(ax=ax, y='latitude', label='Large-scale precipitation fraction')
#ax.legend()
ax.set_title('')
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
ax.set_yticklabels(['90°S', '60°S', '30°S', '0', '30°N', '60°N', '90°N'])
ax.set_ylabel('')
ax.set_xlabel('Convective precipitation fraction')
plt.savefig(plot_path + f'precipitation_fractions_{plot_timestring}.png', dpi=300, bbox_inches='tight')



