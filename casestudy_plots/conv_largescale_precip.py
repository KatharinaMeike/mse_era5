# Compare the horizontal distribution of convective and large-scale precipitation
# with the MSE tendencies due to convection and turbulence.
#%%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import sys
sys.path.append('..')

import mse_budget_centered_diff as mseb
plot_path = '/project2/tas1/katharinah/mse/precip/plots/'

timestep = pd.Timestamp('2005-06-02 01:00')

average_hours = 23

# Read convective and large-scale precipitation from grib file
ds_single = xr.open_dataset('/project2/tas1/katharinah/ERA5/hourly_rad_heat/era5_single_level_fc_20050601',engine='cfgrib')

# Select time in single-level dataset
init_time, step = mseb.get_forecast_time(timestep)
ds_tb = ds_single.sel(time=init_time, step=step)

if average_hours > 0:
    for h in range(1,average_hours+1):
        init_time_h, step_h = mseb.get_forecast_time(timestep + pd.Timedelta(hours=h))
        ds_tb_h = ds_single.sel(time=init_time_h, step=step_h)
        ds_tb = ds_tb + ds_tb_h
    ds_tb = ds_tb / (average_hours + 1)

# Read 3D MSE tendency due to convection and turbulence from mse_budget output
output_path = '/project2/tas1/katharinah/ERA5/mse_output/'
ds_conv = xr.open_dataset(output_path+'mse_tend_conv__2005060300.nc')

#%%
# Plot comparison
fig, axes = plt.subplots(3,2, figsize=(12,7),constrained_layout=True)
axes = axes.flatten()
ax = axes[0]
# Total precipitation
(ds_tb.avg_cpr + ds_tb.avg_lsprate).plot(ax=ax,vmin=0,vmax=3e-4,cmap="Blues")
ax.set_title('Total precipitation rate')

# Convective precipitation
ax = axes[2]
ds_tb.avg_cpr.plot(ax=ax,vmin=0,vmax=3e-4,cmap="Blues",cbar_kwargs={"label":"m/s"})
ax.set_title('Convective precipitation rate')

# Large-scale precipitation
ax = axes[1]
ds_tb.avg_lsprate.plot(ax=ax,vmin=0,vmax=3e-4,cmap="Blues",cbar_kwargs={"label":"m/s"})
ax.set_title('Large-scale precipitation rate')

# MSE tendency at 500 hPa
ax = axes[3]
ds_conv.conv.sel(plev=500,method='nearest').plot(ax=ax,robust=True,cmap="RdBu_r",cbar_kwargs={"label":"W/kg"})
ax.set_title('MSE tendency due to convection and turbulence at 500 hPa')

# MSE tendency at 600 hPa
ax = axes[4]
ds_conv.conv.sel(plev=600,method='nearest').plot(ax=ax,robust=True,cmap="RdBu_r",cbar_kwargs={"label":"W/kg"})
ax.set_title('MSE tendency due to convection and turbulence at 600 hPa')

# MSE tendency at 400 hPa
ax = axes[5]
ds_conv.conv.sel(plev=400,method='nearest').plot(ax=ax,robust=True,cmap="RdBu_r",cbar_kwargs={"label":"W/kg"})
ax.set_title('MSE tendency due to convection and turbulence at 400 hPa')

for ax in axes:
    ax.grid()

plt.savefig(plot_path+'casestudy_20050603_conv_largescale_precip.png',dpi=300,bbox_inches='tight')

#%%
# Is there a quantitative relationship between precipitation rate and MSE tendency?
# Scatter plots. Each point is a grid cell.
fig, ax = plt.subplots(figsize=(6,6))
precip_rate = (ds_tb.avg_cpr + ds_tb.avg_lsprate).values.flatten()
mse_tend_500 = ds_conv.conv.sel(plev=500,method='nearest').values.flatten()
ax.scatter(precip_rate, mse_tend_500, alpha=0.1)
ax.set_xlabel('Total precipitation rate (m/s)')
ax.set_ylabel('MSE tendency due to convection and turbulence at 500 hPa (W/kg)')
ax.set_title('Scatter plot of total precipitation rate vs. MSE tendency at 500 hPa')

#%%
# 2D histogram: precipitation rate vs MSE tendency
precip_rate = (ds_tb.avg_cpr).values.flatten()# + ds_tb.avg_lsprate).values.flatten()
mse_tend_500 = ds_conv.conv.sel(plev=500, method='nearest').values.flatten()

valid = np.isfinite(precip_rate) & np.isfinite(mse_tend_500)
precip_rate = precip_rate[valid]
mse_tend_500 = mse_tend_500[valid]

precip_bins = np.linspace(0, 3e-4, 50)
mse_bins = np.linspace(np.nanpercentile(mse_tend_500, 1),
                       np.nanpercentile(mse_tend_500, 99), 50)

hist2d, precip_edges, mse_edges = np.histogram2d(
    precip_rate, mse_tend_500, bins=[precip_bins, mse_bins]
)
#%%
# Optional: plot the 2D histogram
fig, ax = plt.subplots(figsize=(6, 5))
pcm = ax.pcolormesh(precip_edges, mse_edges, np.log(hist2d.T), cmap="viridis")
fig.colorbar(pcm, ax=ax, label="Count")
ax.set_xlabel("Total precipitation rate (m/s)")
ax.set_ylabel("MSE tendency at 500 hPa (W/kg)")
ax.set_title("2D histogram of precipitation rate vs MSE tendency")
