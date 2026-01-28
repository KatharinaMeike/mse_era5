# MSE budget with parametrized tendencies
# using horizontal derivatives with spherical harmonics
#%%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import numba
import spharm

R_D = 287.0597 # Gas constant for dry air in J/kg/K as in the IFS documentation
cp = 3.5 * R_D # J/kg/K
lh = 2.5008e6 # Latent heat of vaporization in J/kg as in the IFS documentation
r_earth = 6371229.0 # Radius of the earth in m as in the IFS documentation

top = 0.1

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
            dlog_p = np.log(ph_levplusone / top)
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

"""# Parallelized function to interpolate from model levels to pressure levels in log-pressure
@numba.njit(parallel=True)
def interpolate_velocity(ds_var, psurf_latlon, lev_tb):
    var_interp = np.empty((len(lev_tb),ds_var.shape[1], ds_var.shape[2]),dtype='float64')
    for i_lat in numba.prange(ds_var.shape[1]): 
        for i_lon in numba.prange(ds_var.shape[2]):
            p_half = aa + bb * psurf_latlon[i_lat, i_lon]
            phyb_full = 0.5 * (p_half[1:]+p_half[:-1])
            var_hyb = ds_var[:, i_lat, i_lon]
            for i, ilev in enumerate(lev_tb):
                ibelow = np.searchsorted(phyb_full,ilev) # Index of hybrid level below ilev
                iabove = ibelow - 1
                if iabove == -1: # Extrapolate above the top full level
                    # Introduce mock top level (top)
                    # Value at mock top level is linearly extrapolated from the two highest full levels.
                    var_top = (var_hyb[0] * np.log(phyb_full[1]/top) + var_hyb[1] * np.log(top/phyb_full[0]))\
                    / np.log(phyb_full[1]/phyb_full[0])
                    var_interp[i, i_lat, i_lon] = ((np.log(ilev)-np.log(phyb_full[0]))*(np.log(ilev)-np.log(phyb_full[1]))*var_top)/((np.log(top)-np.log(phyb_full[0]))*(np.log(top)-np.log(phyb_full[1]))) +\
                          ((np.log(ilev)-np.log(top))*(np.log(ilev)-np.log(phyb_full[1]))* var_hyb[0])/((np.log(phyb_full[0])-np.log(top))*(np.log(phyb_full[0])-np.log(phyb_full[1]))) +\
                          ((np.log(ilev)-np.log(top))*(np.log(ilev)-np.log(phyb_full[0]))*var_hyb[1])/((np.log(phyb_full[1])-np.log(top))*(np.log(phyb_full[1])-np.log(phyb_full[0])))
                elif (ibelow == 1 and iabove == 0): # Quadratic interpolation between top and second-highest level.
                    var_interp[i, i_lat, i_lon] = ((np.log(ilev)-np.log(phyb_full[1]))*(np.log(ilev)-np.log(phyb_full[2]))*var_hyb[0])/((np.log(phyb_full[0])-np.log(phyb_full[1]))*(np.log(phyb_full[0])-np.log(phyb_full[2]))) +\
                          ((np.log(ilev)-np.log(phyb_full[0]))*(np.log(ilev)-np.log(phyb_full[2]))*var_hyb[1])/((np.log(phyb_full[1])-np.log(phyb_full[0]))*(np.log(phyb_full[1])-np.log(phyb_full[2]))) +\
                          ((np.log(ilev)-np.log(phyb_full[0]))*(np.log(ilev)-np.log(phyb_full[1]))*var_hyb[2])/((np.log(phyb_full[2])-np.log(phyb_full[0]))*(np.log(phyb_full[2])-np.log(phyb_full[1])))
                elif ibelow == len(phyb_full): # Extrapolation below bottom level with constant
                    var_interp[i, i_lat, i_lon] = var_hyb[iabove]
                else: # Linear interpolation in log-pressure
                    var_interp[i, i_lat, i_lon] = (var_hyb[iabove] * np.log(phyb_full[ibelow]/ilev) + var_hyb[ibelow] * np.log(ilev/phyb_full[iabove]))\
                    / np.log(phyb_full[ibelow]/phyb_full[iabove])
    return var_interp"""


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
                if iabove == -1: # Extrapolate above the top full level
                    # Introduce mock top level (top)
                    # Value at mock top level is linearly extrapolated from the two highest full levels.
                    var_top = (var_hyb[0] * np.log(phyb_full[1]/top) + var_hyb[1] * np.log(top/phyb_full[0]))\
                    / np.log(phyb_full[1]/phyb_full[0])
                    var_interp[i, i_lat, i_lon] = ((np.log(ilev)-np.log(phyb_full[0]))*(np.log(ilev)-np.log(phyb_full[1]))*var_top)/((np.log(top)-np.log(phyb_full[0]))*(np.log(top)-np.log(phyb_full[1]))) +\
                          ((np.log(ilev)-np.log(top))*(np.log(ilev)-np.log(phyb_full[1]))* var_hyb[0])/((np.log(phyb_full[0])-np.log(top))*(np.log(phyb_full[0])-np.log(phyb_full[1]))) +\
                          ((np.log(ilev)-np.log(top))*(np.log(ilev)-np.log(phyb_full[0]))*var_hyb[1])/((np.log(phyb_full[1])-np.log(top))*(np.log(phyb_full[1])-np.log(phyb_full[0])))
                elif (ibelow == 1 and iabove == 0): # Quadratic interpolation between top and second-highest level.
                    var_interp[i, i_lat, i_lon] = ((np.log(ilev)-np.log(phyb_full[1]))*(np.log(ilev)-np.log(phyb_full[2]))*var_hyb[0])/((np.log(phyb_full[0])-np.log(phyb_full[1]))*(np.log(phyb_full[0])-np.log(phyb_full[2]))) +\
                          ((np.log(ilev)-np.log(phyb_full[0]))*(np.log(ilev)-np.log(phyb_full[2]))*var_hyb[1])/((np.log(phyb_full[1])-np.log(phyb_full[0]))*(np.log(phyb_full[1])-np.log(phyb_full[2]))) +\
                          ((np.log(ilev)-np.log(phyb_full[0]))*(np.log(ilev)-np.log(phyb_full[1]))*var_hyb[2])/((np.log(phyb_full[2])-np.log(phyb_full[0]))*(np.log(phyb_full[2])-np.log(phyb_full[1])))
                elif ibelow == len(phyb_full): # Extrapolation below bottom level with constant
                    var_interp[i, i_lat, i_lon] = var_hyb[iabove]
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

# MSE flux in x direction u*m
zonal_flux = (mse_1_pl + mse_2_pl) * (u_1_pl + u_2_pl) / 4.0

# MSE flux in y direction v*m 
merid_flux = (mse_1_pl + mse_2_pl) * (v_1_pl + v_2_pl) / 4.0

# MSE flux in z direction w*m
vertical_flux = (mse_1_pl + mse_2_pl) * (w_1_pl + w_2_pl) / 4.0


#%%
# Horizontal divergence of MSE flux with spherical harmonics
u_pl = 0.5 * (u_1_pl + u_2_pl)
v_pl = 0.5 * (v_1_pl + v_2_pl)
w_pl = 0.5 * (w_1_pl + w_2_pl)
x=spharm.Spharmt(1440,721,rsphere=r_earth,gridtype='regular',legfunc='stored')
flux_divergence = np.zeros_like(u_pl)
for i_lev in range(len(plev)):
    vrtspec, divspec = x.getvrtdivspec(zonal_flux[i_lev], merid_flux[i_lev])
    flux_divergence[i_lev] = x.spectogrd(divspec)


# Vertical divergence of MSE flux
dp_vertical_flux = np.gradient(vertical_flux, plev*100.0, axis=0)

#%%
# looks not very bad
vminmax = 1.0
plt.imshow(np.mean((flux_divergence + dp_vertical_flux - conv_tend_pl - rad_tend_pl + mse_diff_pl)[:,:,300:400],axis=-1),vmin=-vminmax,vmax=vminmax)
plt.colorbar()
plt.show()

#%%
# Temperature budget in flux form
temperature_1_pl = cp * interpolate_to_pressure(temperature_1, phyb_half_1[-1], plev*100.0)
temperature_2_pl = cp * interpolate_to_pressure(temperature_2, phyb_half_2[-1], plev*100.0)
temperature_sum = (temperature_1_pl + temperature_2_pl)

# MSE flux in x direction u*m
zonal_temperature_flux = (temperature_1_pl + temperature_2_pl) * (u_1_pl + u_2_pl) / 4.0

# MSE flux in y direction v*m 
merid_temperature_flux = (temperature_1_pl + temperature_2_pl) * (v_1_pl + v_2_pl) / 4.0

# MSE flux in z direction w*m
vertical_temperature_flux = (temperature_1_pl + temperature_2_pl) * (w_1_pl + w_2_pl) / 4.0

x=spharm.Spharmt(1440,721,rsphere=r_earth,gridtype='regular',legfunc='stored')
temperature_flux_divergence = np.zeros_like(u_pl)
for i_lev in range(len(plev)):
    vrtspec, divspec = x.getvrtdivspec(zonal_temperature_flux[i_lev], merid_temperature_flux[i_lev])
    temperature_flux_divergence[i_lev] = x.spectogrd(divspec)

# Vertical divergence of MSE flux
dp_vertical_temperature_flux = np.gradient(vertical_temperature_flux, plev*100.0, axis=0)

temperature_diff = (temperature_2_pl - temperature_1_pl) / 3600.0

temperature_param = cp * ds_tend.isel(valid_time=timestep).avg_ttpm.values

temperature_param_pl = interpolate_to_pressure(temperature_param, phyb_half_1[-1], plev*100.0)

geopot_1_pl = interpolate_to_pressure(geopot_1, phyb_half_1[-1], plev*100.0)
geopot_2_pl = interpolate_to_pressure(geopot_2, phyb_half_2[-1], plev*100.0)
dp_geopot = 0.25 * (w_1_pl + w_2_pl) * np.gradient((geopot_1_pl+geopot_2_pl), plev*100.0, axis=0)

rhs = temperature_param_pl - temperature_flux_divergence - dp_vertical_temperature_flux - dp_geopot

