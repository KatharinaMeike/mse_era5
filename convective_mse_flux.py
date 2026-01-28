# MSE tendency due to parametrizations affecting temperature and moisture
#%%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import numba

R_D = 287.0597 # Gas constant for dry air in J/kg/K as in the IFS documentation
cp = 3.5 * R_D # J/kg/K
lh = 2.5008e6 # Latent heat of vaporization in J/kg as in the IFS documentation
r_earth = 6371229.0 # Radius of the earth in m as in the IFS documentation

data = 'scratch2'

if data == 'scratch':
    data_path = '/scratch/midway3/katharinah/ERA5/ml/2005-06-02/'
    ml_path = '/project2/tas1/katharinah/ERA5/ml/'
elif data == 'scratch2':
    data_path = '/scratch/midway3/katharinah/ERA5/ml/2005-06-01/'
    ml_path = '/project2/tas1/katharinah/ERA5/ml/'
else:
    data_path = '/project2/tas1/katharinah/ERA5/ml/'
    ml_path = data_path

# Read parameters for model levels 
mlev = np.genfromtxt(ml_path+'model_levels_L137.dat')
aa = mlev[:,1].astype('float64')
bb = mlev[:,2].astype('float64')

plev = mlev[1:,4].astype('float64')

# Compute geopotential from temperature, based on the code from https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height
def calc_geopot(temperature, specif_hum, phyb_half, surface_geopot):
    virt_temp = temperature * (1. + 0.609133 * specif_hum)
    geopot = np.zeros_like(virt_temp)
    z_h = surface_geopot

    # Integrate from bottom of the atmosphere upward
    for i in range(geopot.shape[0]-1,-1,-1):
        ph_lev = phyb_half[i]
        ph_levplusone = phyb_half[i+1]
        if i == 0:
            dlog_p = np.log(ph_levplusone / 0.1)
            alpha = np.log(2)
        else:
            dlog_p = np.log(ph_levplusone / ph_lev)
            alpha = 1. - ((ph_lev / (ph_levplusone - ph_lev)) * dlog_p)

        # z_f is the geopotential of this full level
        # integrate from previous (lower) half-level z_h to the
        # full level
        geopot[i] = z_h + (virt_temp[i] * R_D * alpha)

        # z_h is the geopotential of 'half-levels'
        # integrate z_h to next half level
        z_h = z_h + (virt_temp[i] * R_D * dlog_p)

    return geopot

# Parallelized function to interpolate from model levels to pressure levels in log-pressure
@numba.njit(parallel=True)
def interpolate_to_pressure(ds_var, psurf_latlon, lev_tb):
    var_interp = np.empty((len(lev_tb),ds_var.shape[1], ds_var.shape[2]),dtype='float64')
    for i_lat in numba.prange(ds_var.shape[1]): 
        for i_lon in numba.prange(ds_var.shape[2]):
            p_half = aa + bb * psurf_latlon[i_lat, i_lon]
            phyb_full = 0.5 * (p_half[1:]+p_half[:-1])
            var_hyb = ds_var[:, i_lat, i_lon]
            for i, ilev in enumerate(lev_tb):
                ibelow = np.searchsorted(phyb_full,ilev) # Index of hybrid level below ilev
                iabove = ibelow - 1
                if iabove == -1: # Set to NaN above the top level
                    var_interp[i, i_lat, i_lon] = np.nan
                elif ibelow == len(phyb_full): # Set to NaN below the lowest level
                    var_interp[i, i_lat, i_lon] = np.nan
                else: # Linear interpolation in log-pressure
                    var_interp[i, i_lat, i_lon] = (var_hyb[iabove] * np.log(phyb_full[ibelow]/ilev) + var_hyb[ibelow] * np.log(ilev/phyb_full[iabove]))\
                    / np.log(phyb_full[ibelow]/phyb_full[iabove])
    return var_interp

#%%

if data == 'scratch':
    # Read tendencies on model levels
    ds_tend = xr.open_dataset(data_path+'era5_ml_3D_20050602.nc')

    # Read surface pressure and surface geopotential
    ds_ps = xr.open_dataset(data_path+'era5_ml_20050602.nc').isel(model_level=0)

    # Read u and v wind
    ds_wind = xr.open_dataset(data_path+'era5_ml_3D_20050602.nc')

    # Read other variables
    ds_variab = xr.open_dataset(data_path+'era5_ml_3D_20050602.nc')
else:
    # Read tendencies on model levels
    ds_tend = xr.open_dataset(data_path+'era5_model_level_tend.nc')

    # Read surface pressure and surface geopotential
    ds_ps = xr.open_dataset(data_path+'era5_model_level_var.nc').isel(model_level=0)

    # Read u and v wind
    ds_wind = xr.open_dataset(data_path+'era5_model_level_wind.nc')

    # Read other variables
    ds_variab = xr.open_dataset(data_path+'era5_model_level_var2.nc')

# Variables at first timestep

timestep = 1

surface_geopot_1 = ds_ps.isel(valid_time=timestep).z.values # Surface geopotential is lowest half-level
temperature_1 = ds_variab.isel(valid_time=timestep).t.values
specif_hum_1 = ds_variab.isel(valid_time=timestep).q.values
u_1 = ds_wind.isel(valid_time=timestep).u.values
v_1 = ds_wind.isel(valid_time=timestep).v.values
w_1 = ds_variab.isel(valid_time=timestep).w.values

# Pressure on half-levels
phyb_half_1 = aa[:,np.newaxis,np.newaxis] +\
            bb[:,np.newaxis,np.newaxis] * np.exp(ds_ps.isel(valid_time=timestep).lnsp.values)[np.newaxis,:,:]

# Distance between pressure levels
pdist_1 = phyb_half_1[1:] - phyb_half_1[:-1]

geopot_1 = calc_geopot(temperature_1, specif_hum_1, phyb_half_1, surface_geopot_1)

# MSE
mse_1 = temperature_1 * cp + specif_hum_1 * lh + geopot_1
#mse_1 = temperature_1 * cp + specif_hum_1 * lh# + geopot_1


# ------------------------------
# Variables at second timestep
timestep = 2

surface_geopot_2 = ds_ps.isel(valid_time=timestep).z.values # Surface geopotential is lowest half-level
temperature_2 = ds_variab.isel(valid_time=timestep).t.values
specif_hum_2 = ds_variab.isel(valid_time=timestep).q.values
u_2 = ds_wind.isel(valid_time=timestep).u.values
v_2 = ds_wind.isel(valid_time=timestep).v.values
w_2 = ds_variab.isel(valid_time=timestep).w.values

# Pressure on half-levels
phyb_half_2 = aa[:,np.newaxis,np.newaxis] +\
            bb[:,np.newaxis,np.newaxis] * np.exp(ds_ps.isel(valid_time=timestep).lnsp.values)[np.newaxis,:,:]

geopot_2 = calc_geopot(temperature_2, specif_hum_2, phyb_half_2, surface_geopot_2)

# MSE
mse_2 = temperature_2 * cp + specif_hum_2 * lh + geopot_2
#mse_2 = temperature_2 * cp + specif_hum_2 * lh# + geopot_2

# Tendencies due to parametrizations are given at the end of the hour for which they are valid
# cp * d_t T_nonrad + lh * d_t q
conv_tend = cp * (ds_tend.isel(valid_time=timestep).avg_ttpm.values -\
                ds_tend.isel(valid_time=timestep).avg_ttswr.values -\
                ds_tend.isel(valid_time=timestep).avg_ttlwr.values) +\
                lh * ds_tend.isel(valid_time=timestep).avg_qtpm.values

# Radiative processes
rad_tend = cp * (ds_tend.isel(valid_time=timestep).avg_ttswr.values +\
                ds_tend.isel(valid_time=timestep).avg_ttlwr.values)

#%%
# Interpolate from model levels to pressure levels in log-pressure
mse_1_pl = interpolate_to_pressure(mse_1, phyb_half_1[-1], plev*100.0)
mse_2_pl = interpolate_to_pressure(mse_2, phyb_half_2[-1], plev*100.0)

u_1_pl = interpolate_to_pressure(u_1, phyb_half_1[-1], plev*100.0)
u_2_pl = interpolate_to_pressure(u_2, phyb_half_2[-1], plev*100.0)

v_1_pl = interpolate_to_pressure(v_1, phyb_half_1[-1], plev*100.0)
v_2_pl = interpolate_to_pressure(v_2, phyb_half_2[-1], plev*100.0)

w_1_pl = interpolate_to_pressure(w_1, phyb_half_1[-1], plev*100.0)
w_2_pl = interpolate_to_pressure(w_2, phyb_half_2[-1], plev*100.0)

# For tendencies, use half-levels at the end of the hour
conv_tend_pl = interpolate_to_pressure(conv_tend, phyb_half_2[-1], plev*100.0)
rad_tend_pl = interpolate_to_pressure(rad_tend, phyb_half_2[-1], plev*100.0)

# Difference in MSE between the two timesteps per second
mse_diff_pl = (mse_2_pl - mse_1_pl) / 3600.0

x_spacing = np.deg2rad(0.25) * np.cos(np.deg2rad(ds_ps.latitude.values)) * r_earth # Distance in meters between longitudinal grid points for each latitude
y_spacing = np.deg2rad(0.25) * r_earth

#%%
# Check whether the continuity equation works well with centered differences
u_pl = 0.5 * (u_1_pl + u_2_pl)
v_pl = 0.5 * (v_1_pl + v_2_pl)
w_pl = 0.5 * (w_1_pl + w_2_pl)
dx_u = np.zeros_like(u_1_pl)
for i_lat in range(len(x_spacing)):
    dx_u[:,i_lat,:] = np.gradient(u_pl[:,i_lat], x_spacing[i_lat], axis=1)

dy_v = np.gradient(v_pl * np.cos(np.deg2rad(ds_ps.latitude.values))[np.newaxis,:,np.newaxis], y_spacing, axis=1) / np.cos(np.deg2rad(ds_ps.latitude.values))[np.newaxis,:,np.newaxis]

dp_w = np.gradient(w_pl, plev*100.0, axis=0)

conti = dx_u - dy_v + dp_w

#%%
# MSE flux in x direction u*m
zonal_flux = (mse_1_pl + mse_2_pl) * (u_1_pl + u_2_pl) / 4.0

# MSE flux in y direction v*m 
merid_flux = (mse_1_pl + mse_2_pl) * (v_1_pl + v_2_pl) / 4.0

# MSE flux in z direction w*m
vertical_flux = (mse_1_pl + mse_2_pl) * (w_1_pl + w_2_pl) / 4.0


# Gradients of fluxes
dx_zonal_flux = np.zeros_like(zonal_flux)
for i_lat in range(len(x_spacing)):
    dx_zonal_flux[:,i_lat,:] = np.gradient(zonal_flux[:,i_lat], x_spacing[i_lat], axis=1)

dy_merid_flux = np.gradient(merid_flux * np.cos(np.deg2rad(ds_ps.latitude.values))[np.newaxis,:,np.newaxis], y_spacing, axis=1) / np.cos(np.deg2rad(ds_ps.latitude.values))[np.newaxis,:,np.newaxis]

dp_vertical_flux = np.gradient(vertical_flux, plev*100.0, axis=0)

#%%
# Sum of all tendencies
rhs = conv_tend_pl + rad_tend_pl - (dx_zonal_flux + dp_vertical_flux - dy_merid_flux)

#%%
# Are zonal and meridional flux of geopotential really small?
geopot_1_pl = interpolate_to_pressure(geopot_1, phyb_half_1[-1], plev*100.0)
geopot_2_pl = interpolate_to_pressure(geopot_2, phyb_half_2[-1], plev*100.0)
# MSE flux in x direction u*m
zonal_geopot_flux = (geopot_1_pl + geopot_2_pl) * (u_1_pl + u_2_pl) / 4.0

# MSE flux in y direction v*m 
merid_geopot_flux = (geopot_1_pl + geopot_2_pl) * (v_1_pl + v_2_pl) / 4.0

#%%
dx_zonal_geopot_flux = np.zeros_like(zonal_geopot_flux)
for i_lat in range(len(x_spacing)):
    dx_zonal_geopot_flux[:,i_lat,:] = np.gradient(zonal_geopot_flux[:,i_lat], x_spacing[i_lat], axis=1)

dy_merid_geopot_flux = np.gradient(merid_geopot_flux * np.cos(np.deg2rad(ds_ps.latitude.values))[np.newaxis,:,np.newaxis], y_spacing, axis=1) / np.cos(np.deg2rad(ds_ps.latitude.values))[np.newaxis,:,np.newaxis]

# The divergence of the geopotential flux does not seem to be small

geopot_tendency = (geopot_2_pl - geopot_1_pl) / 3600.0 # this is small

#%%
# Fluxes of c_p T
temperature_1_pl = cp * interpolate_to_pressure(temperature_1, phyb_half_1[-1], plev*100.0)
temperature_2_pl = cp * interpolate_to_pressure(temperature_2, phyb_half_2[-1], plev*100.0)



zonal_temperature_flux = (temperature_1_pl + temperature_2_pl) * (u_1_pl + u_2_pl) / 4.0

merid_temperature_flux = (temperature_1_pl + temperature_2_pl) * (v_1_pl + v_2_pl) / 4.0

vertical_temperature_flux = (temperature_1_pl + temperature_2_pl) * (w_1_pl + w_2_pl) / 4.0

temperature_param = cp * ds_tend.isel(valid_time=timestep).avg_ttpm.values

temperature_param_pl = interpolate_to_pressure(temperature_param, phyb_half_1[-1], plev*100.0)

dp_geopot = 0.25 * (w_1_pl + w_2_pl) * np.gradient((geopot_1_pl+geopot_2_pl), plev*100.0, axis=0)


#%%

dx_zonal_temperature_flux = np.zeros_like(zonal_temperature_flux)
for i_lat in range(len(x_spacing)):
    dx_zonal_temperature_flux[:,i_lat,:] = np.gradient(zonal_temperature_flux[:,i_lat], x_spacing[i_lat], axis=1)

dy_merid_temperature_flux = np.gradient(merid_temperature_flux * np.cos(np.deg2rad(ds_ps.latitude.values))[np.newaxis,:,np.newaxis], y_spacing, axis=1) / np.cos(np.deg2rad(ds_ps.latitude.values))[np.newaxis,:,np.newaxis]

dp_vertical_temperature_flux = np.gradient(vertical_temperature_flux, plev*100.0, axis=0)

#%%
temperature_sum = (temperature_1_pl + temperature_2_pl)
dx_temperature = np.zeros_like(zonal_temperature_flux)
for i_lat in range(len(x_spacing)):
    dx_temperature[:,i_lat,:] = np.gradient(temperature_sum[:,i_lat], x_spacing[i_lat], axis=1)
dx_temperature = dx_temperature * 0.25 * (u_1_pl + u_2_pl)

dy_temperature = 0.25 * (v_1_pl + v_2_pl) * np.gradient((temperature_1_pl + temperature_2_pl), y_spacing, axis=1)

dp_temperature = 0.25 * (w_1_pl + w_2_pl) * np.gradient((temperature_1_pl + temperature_2_pl), plev*100.0, axis=0)

temperature_diff = (temperature_2_pl - temperature_1_pl) / 3600.0

rhs = temperature_param_pl - dx_zonal_temperature_flux + dy_merid_temperature_flux - dp_vertical_temperature_flux - dp_geopot

#%%
# Seems to be small
vminmax=1
plt.imshow(np.mean((dx_temperature - dy_temperature + dp_temperature - temperature_param_pl + dp_geopot + temperature_diff)[:,:,300:400],axis=-1),vmin=-vminmax,vmax=vminmax)
plt.colorbar()
plt.show()

#%%
# Should be equivalent, but is not (probably because the discretizaiton does not eliminate continuity)
vminmax=1
plt.imshow(np.mean((dx_zonal_temperature_flux - dy_merid_temperature_flux + dp_vertical_temperature_flux - temperature_param_pl + dp_geopot + temperature_diff)[:,:,300:400],axis=-1),vmin=-vminmax,vmax=vminmax)
plt.colorbar()
plt.show()

#%%
# Subtracting continuity makes it small
vminmax=1
plt.imshow(np.mean((conti * (-0.5) * temperature_sum + dx_zonal_temperature_flux - dy_merid_temperature_flux + dp_vertical_temperature_flux - temperature_param_pl + dp_geopot + temperature_diff)[:,:,300:400],axis=-1),vmin=-vminmax,vmax=vminmax)
plt.colorbar()
plt.show()


#%%
# Divergence with spherical harmonics
import spharm
x=spharm.Spharmt(1440,721,rsphere=r_earth,gridtype='regular',legfunc='stored')
divergence = np.zeros_like(u_pl)
for i_lev in range(len(plev)):
    vrtspec, divspec = x.getvrtdivspec(u_pl[i_lev], v_pl[i_lev])
    divergence[i_lev] = x.spectogrd(divspec)

conti_div = divergence + dp_w



#%%

# MSE tendency cp d_t T + L d_t q
#mse_tend = cp * (ds_tend.avg_ttpm - ds_tend.avg_ttswr - ds_tend.avg_ttlwr) + lh * ds_tend.avg_qtpm

#%%
# Integrate MSE over the half-levels
#mse_integrated = np.zeros_like(mse_tend.values)
#mse_integrated[0] = mse_tend.isel(model_level=0).values * pdist[0]
#for i, lev in enumerate(range(len(pdist)-1)):
#    mse_integrated[i+1] = mse_integrated[i] + mse_tend.isel(model_level=i+1).values * pdist[i+1]

#msetend_da = xr.Dataset({'conv':(('half_level','latitude','longitude'),mse_integrated)},coords={'half_level':np.arange(1,len(pdist)+1),'latitude':ds_ps.latitude.values,'longitude':ds_ps.longitude.values})

#%%



