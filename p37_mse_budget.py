# Compute quantities for MSE budget for data on 37 pressure levels
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

import mse_budget_centered_diff as mseb
output_path = '/project2/tas1/katharinah/ERA5/mse_output/'
for timestep in pd.date_range('2005-03-02 00:00','2005-03-03 00:00',freq='1d'):
    bud = mseb.mse_budget(config_file='mse_budget.ini', timestep=timestep, time_interval='1 day', budget='daily_P37_varp')
    bud.compute_flux_divergences()

    levlat_output = []
    latlon_output = []

    # Tendency of MSE
    msetend_levlat = np.nanmean(bud.mse_tendency, axis=-1) # Zonal mean
    msetend_latlon = bud.compute_vertical_integral(bud.mse_tendency)
    levlat_output.append(bud.to_xarray_levlat(msetend_levlat, var_name='mse_tendency'))
    latlon_output.append(bud.to_xarray_latlon(msetend_latlon, var_name='mse_tendency'))

    # Zonal advection of MSE
    zonal_adv = bud.compute_zonal_advection(bud.mse)
    zonal_adv_levlat = np.nanmean(zonal_adv, axis=-1) # Zonal mean
    zonal_adv_latlon = bud.compute_vertical_integral(zonal_adv)
    levlat_output.append(bud.to_xarray_levlat(zonal_adv_levlat, var_name='zonal_adv_mse'))
    latlon_output.append(bud.to_xarray_latlon(zonal_adv_latlon, var_name='zonal_adv_mse'))
    zonal_adv = None

    # Meridional advection of MSE
    meridional_adv = bud.compute_meridional_advection(bud.mse)
    meridional_adv_levlat = np.nanmean(meridional_adv, axis=-1) # Zonal mean
    meridional_adv_latlon = bud.compute_vertical_integral(meridional_adv)
    levlat_output.append(bud.to_xarray_levlat(meridional_adv_levlat, var_name='meridional_adv_mse'))
    latlon_output.append(bud.to_xarray_latlon(meridional_adv_latlon, var_name='meridional_adv_mse'))
    meridional_adv = None

    # Vertical advection of MSE
    vertical_adv = bud.compute_vertical_advection(bud.mse)
    vertical_adv_levlat = np.nanmean(vertical_adv, axis=-1) # Zonal mean
    vertical_adv_latlon = bud.compute_vertical_integral(vertical_adv)
    levlat_output.append(bud.to_xarray_levlat(vertical_adv_levlat, var_name='vertical_adv_mse'))
    latlon_output.append(bud.to_xarray_latlon(vertical_adv_latlon, var_name='vertical_adv_mse'))
    vertical_adv = None

    # Zonal MSE flux
    zonal_flux_levlat = np.nanmean(bud.zonal_flux, axis=-1) # Zonal mean
    zonal_flux_latlon = bud.compute_vertical_integral(bud.zonal_flux)
    levlat_output.append(bud.to_xarray_levlat(zonal_flux_levlat, var_name='zonal_mse_flux'))
    latlon_output.append(bud.to_xarray_latlon(zonal_flux_latlon, var_name='zonal_mse_flux'))

    # Meridional MSE flux
    meridional_flux_levlat = np.nanmean(bud.merid_flux, axis=-1) # Zonal mean
    meridional_flux_latlon = bud.compute_vertical_integral(bud.merid_flux)
    levlat_output.append(bud.to_xarray_levlat(meridional_flux_levlat, var_name='meridional_mse_flux'))
    latlon_output.append(bud.to_xarray_latlon(meridional_flux_latlon, var_name='meridional_mse_flux'))

    # Vertical MSE flux
    vertical_flux_levlat = np.nanmean(bud.vertical_flux, axis=-1)
    vertical_flux_latlon = bud.compute_vertical_integral(bud.vertical_flux)
    levlat_output.append(bud.to_xarray_levlat(vertical_flux_levlat, var_name='vertical_mse_flux'))
    latlon_output.append(bud.to_xarray_latlon(vertical_flux_latlon, var_name='vertical_mse_flux'))

    """# Divergence of zonal MSE flux
    dx_zonal_flux_levlat = np.nanmean(bud.dx_zonal_flux, axis=-1) # Zonal mean
    dx_zonal_flux_latlon = bud.compute_vertical_integral(bud.dx_zonal_flux)
    levlat_output.append(bud.to_xarray_levlat(dx_zonal_flux_levlat, var_name='dx_zonal_mse_flux'))
    latlon_output.append(bud.to_xarray_latlon(dx_zonal_flux_latlon, var_name='dx_zonal_mse_flux'))

    # Divergence of meridional MSE flux
    dy_meridional_flux_levlat = np.nanmean(bud.dy_merid_flux, axis=-1) # Zonal mean
    dy_meridional_flux_latlon = bud.compute_vertical_integral(bud.dy_merid_flux)
    levlat_output.append(bud.to_xarray_levlat(dy_meridional_flux_levlat, var_name='dy_meridional_mse_flux'))
    latlon_output.append(bud.to_xarray_latlon(dy_meridional_flux_latlon, var_name='dy_meridional_mse_flux'))

    # Divergence of vertical MSE flux
    dp_vertical_flux_levlat = np.nanmean(bud.dp_vertical_flux, axis=-1) # Zonal mean
    dp_vertical_flux_latlon = bud.compute_vertical_integral(bud.dp_vertical_flux)
    levlat_output.append(bud.to_xarray_levlat(dp_vertical_flux_levlat, var_name='dp_vertical_mse_flux'))
    latlon_output.append(bud.to_xarray_latlon(dp_vertical_flux_latlon, var_name='dp_vertical_mse_flux'))"""

    output_filename = f"mse_daily_P37_varp_levlat"
    xr.merge(levlat_output).to_netcdf(output_path + output_filename + timestep.strftime('_%Y%m%d%H') + '.nc')

    output_filename = f"mse_daily_P37_varp_latlon"
    xr.merge(latlon_output).to_netcdf(output_path + output_filename + timestep.strftime('_%Y%m%d%H') + '.nc')
    bud = None
    levlat_output = None
    latlon_output = None
