# Moist static energy budget from ERA5 data using centered differences
#%%
import numpy as np
import numba
import configparser
import xarray as xr
import pandas as pd

R_D = 287.0597 # Gas constant for dry air in J/kg/K as in the IFS documentation
cp = 3.5 * R_D # J/kg/K
lh = 2.5008e6 # Latent heat of vaporization in J/kg as in the IFS documentation
r_earth = 6371229.0 # Radius of the earth in m as in the IFS documentation
psurf_hPa = 1013.25 # Standard surface pressure in hPa
g_earth = 9.80665 # Gravity in m/s² as in the IFS documentation

#%%


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
def interpolate_to_pressure(ds_var, psurf_latlon, lev_tb, aa, bb):
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

def get_forecast_time(timestep):
        """Determine initialization time ("init_time") and lead time ("step") from timestep and previous_timestep."""
        if timestep.hour > 18:
            init_time = timestep.normalize() + pd.Timedelta('18 hours')
        elif (timestep.hour > 6) and (timestep.hour <= 18):
            init_time = timestep.normalize() + pd.Timedelta('6 hours')
        elif timestep.hour <= 6:
            init_time = timestep.normalize() - pd.Timedelta('6 hours')
        else:
            print('Warning: Timestep hour not recognized for forecast time calculation')
        step = timestep - init_time
        return init_time, step

def get_forecast_time_previous(timestep):
        """Determine initialization time ("init_time") and lead time ("step") from timestep and previous_timestep.
        In contrast to get_forecast_time, this function returns the analysis time, not the 12 hour time of the previous forecast."""
        if timestep.hour >= 18:
            init_time = timestep.normalize() + pd.Timedelta('18 hours')
        elif (timestep.hour >= 6) and (timestep.hour < 18):
            init_time = timestep.normalize() + pd.Timedelta('6 hours')
        elif timestep.hour < 6:
            init_time = timestep.normalize() - pd.Timedelta('6 hours')
        else:
            print('Warning: Timestep hour not recognized for forecast time calculation')
        step = timestep - init_time
        return init_time, step

class mse_budget:
    """Class for MSE budget analysis from netCDF files."""
    
    def __init__(self, config_file='mse_budget.ini', timestep=None, time_interval=None,\
                  budget='hourly_P137'):
        """
        Initialize MSE budget object.
        
        Parameters
        ----------
        config_file : str
            Path to configuration file (default: mse_budget.ini)
        timestep : str
            Timestep string, e.g., '2005-06-01 7:00'
        time_interval : str
            Time interval string, e.g., '1 hour'
        """
        self.config = self._read_config(config_file)
        self.timestep = pd.Timestamp(timestep) if timestep else None
        self.time_interval = time_interval
        self.previous_timestep = self.timestep - pd.Timedelta(self.time_interval)  if timestep else None
        self.current_init_time = None
        self.current_step = None
        self.previous_init_time = None
        self.previous_step = None
        self.u_current_ml = None
        self.u_previous_ml = None
        self.u_current = None
        self.u_previous = None
        self.v_current_ml = None
        self.v_previous_ml = None
        self.v_current = None
        self.v_previous = None
        self.t_current_ml = None
        self.t_previous_ml = None
        self.t_current = None
        self.t_previous = None
        self.q_current_ml = None
        self.q_previous_ml = None
        self.q_current = None
        self.q_previous = None
        self.w_current_ml = None
        self.w_previous_ml = None
        self.w_current = None
        self.w_previous = None
        self.geopot_current_ml = None
        self.geopot_previous_ml = None
        self.geopot_current = None
        self.geopot_previous = None
        self.mse_current = None
        self.mse_previous = None
        self.mse_tendency = None
        self.t_param = None
        self.q_param = None
        self.shortwave = None
        self.longwave = None
        self.mse_tend_sw = None
        self.mse_tend_lw = None
        self.mse_tend_conv = None
        self.mse = None
        self.u = None
        self.v = None
        self.w = None
        self.zonal_flux = None
        self.merid_flux = None
        self.vertical_flux = None
        self.x_spacing = None
        self.y_spacing = None
        self.phyb_half_current = None
        self.phyb_half_previous = None
        self.aa = None
        self.bb = None
        self.plev = None
        self.plev_half = None
        self.pdiff = None
        self.current_datasets = {}
        self.previous_datasets = {}

        if budget == 'hourly_P137':
            self._load_model_levels()
            self._get_forecast_time()
            self.load_data()
            self._extract_variables()
            self._extract_param_tendencies()
            self._compute_phyb_half() 
            self._compute_grid_spacing()           
            self._compute_geopot()            
            self._interpolate_variables_to_pressure()
            self._compute_mse()
            self._compute_mse_tendency()
            self._compute_param_mse_tendencies()
            self._compute_temporal_averages()
            self._compute_mse_fluxes()
        elif budget == 'daily_P137':
            self._load_model_levels()
            self._get_forecast_time()
            self.load_data(tendencies=False)
            self._extract_variables()
            self._compute_phyb_half()
            self._compute_grid_spacing()
            self._compute_geopot()
            self._interpolate_variables_to_pressure()
            self._compute_mse()
            self._compute_mse_tendency()
            self._compute_temporal_averages()
            self._compute_mse_fluxes()
            self.param_mse_tendencies_from_hourly()
            self._compute_param_mse_tendencies()
        elif budget == 'daily_P37':
            self._load_model_levels(plev_in_mlev=False)
            self._get_forecast_time()
            self.load_data(tendencies=False)
            self._extract_variables()
            self._compute_phyb_half()
            self._compute_grid_spacing()
            self._compute_geopot()
            self._interpolate_variables_to_pressure()
            self._compute_mse()
            self._compute_mse_tendency()
            self._compute_temporal_averages()
            self._compute_mse_fluxes()
            self.param_mse_tendencies_from_hourly()
            self._compute_param_mse_tendencies()
        # Read variables from pressure-level data, do not use tendencies due to paramterizations
        elif budget == 'daily_P37_varp':
            self.load_netcdf_plev_data()
            self.extract_plev_variables()
            self._compute_mse()
            self._compute_mse_tendency()
            self._compute_temporal_averages()
            self._compute_mse_fluxes()



    def _get_forecast_time(self):
        """Determine initialization time ("time") and lead time ("step") from timestep and previous_timestep."""
        self.current_init_time, self.current_step = get_forecast_time(self.timestep)
        self.previous_init_time, self.previous_step = get_forecast_time_previous(self.previous_timestep)
    
    def _read_config(self, config_file):
        """Read configuration from ini file."""
        config = configparser.ConfigParser()
        config.read(config_file)
        return config
    
    def _load_model_levels(self, plev_in_mlev = True):
        """Load model level parameters (aa, bb, plev, plev_half) from file."""
        try:
            ml_path = self.config['data'].get('file_model_level', '').strip()
            if not ml_path:
                print("Warning: Model level file not specified in config")
                return
            
            mlev = np.genfromtxt(ml_path)
            self.aa = mlev[:, 1].astype('float64')
            self.bb = mlev[:, 2].astype('float64')
            if plev_in_mlev:
                self.plev = mlev[1:, 4].astype('float64')
            else:
                pl_file = self.config['data'].get('file_pressure_levels', '').strip()
                self.plev = np.genfromtxt(pl_file,delimiter='/',dtype='float64')[::-1]
                print(self.plev)
            
            # Compute plev_half (midpoints between levels, with 0 at the top and psurf_hPa at the bottom)
            midpoints = 0.5 * (self.plev[:-1] + self.plev[1:])
            self.plev_half = np.concatenate([[0.0], midpoints, [psurf_hPa]])
            self.pdiff = self.plev_half[1:] - self.plev_half[:-1]
            
            print(f"Loaded model levels from {ml_path}")
            print(f"  aa shape: {self.aa.shape}")
            print(f"  bb shape: {self.bb.shape}")
            print(f"  plev shape: {self.plev.shape}")
            print(f"  plev_half shape: {self.plev_half.shape}")
            
        except FileNotFoundError as e:
            print(f"Error: Model level file not found - {e}")
        except Exception as e:
            print(f"Error loading model levels - {e}")
    
    def load_data(self, tendencies=True):
        """
        Load netCDF files specified in configuration.
        
        Returns
        -------
        dict
            Dictionary containing loaded xarray datasets
        """
        data_path = self.config['data']['data_path']
        
        # List of file keys in the config
        if tendencies:
            file_keys = ['file_tendencies', 'file_surface', 'file_variables']
        else:
            file_keys = ['file_surface', 'file_variables']
        
        for key in file_keys:
            filename = self.config['data'].get(key, '').strip()
            if filename:  # Only load if filename is specified
                print(f"Loading {filename}...")
                file_time = self.current_init_time.normalize()
                self.current_datasets[key] = xr.open_dataset(data_path + '/' + str(file_time)[:10] \
                        +'/'+ filename + str(file_time.year)+str(file_time.month).zfill(2)+str(file_time.day).zfill(2),engine='cfgrib')
                print(f"  Variables: {list(self.current_datasets[key].data_vars)}")
                print(f"  Dimensions: {dict(self.current_datasets[key].dims)}")
                if key == 'file_surface':
                    self.latitude = np.float64(self.current_datasets[key].latitude.values)
                    self.longitude = np.float64(self.current_datasets[key].longitude.values)


        if (self.previous_init_time.normalize() == self.current_init_time.normalize()):
            self.previous_datasets = self.current_datasets
        else:
            for key in file_keys:
                filename = self.config['data'].get(key, '').strip()
                if filename:  # Only load if filename is specified
                    print(f"Loading {filename}...")
                    file_time = self.previous_init_time.normalize()
                    self.previous_datasets[key] = xr.open_dataset(data_path + '/' + str(file_time)[:10] \
                            +'/'+ filename + str(file_time.year)+str(file_time.month).zfill(2)+str(file_time.day).zfill(2),engine='cfgrib') 
                    print(f"  Variables: {list(self.previous_datasets[key].data_vars)}")
                    print(f"  Dimensions: {dict(self.previous_datasets[key].dims)}")
    
        
        return self.current_datasets, self.previous_datasets
    
    def load_netcdf_plev_data(self,file_string = '.6hrly.nc'):
        """
        Load netCDF data for u, v, t, q, w, z.
        The data should be on the standard 37 pressure levels from ECMWF.
        """
        file_keys = {'file_u', 'file_v', 'file_t', 'file_q', 'file_w', 'file_z'}
        for key in file_keys:
            filename = self.config['data'].get(key, '').strip()
            if filename:  # Only load if filename is specified
                print(f"Loading {filename}...")
                self.current_datasets[key] = xr.open_dataset(filename +\
                    str(self.timestep.year) + '_' + str(self.timestep.month).zfill(2) + file_string)
                self.previous_datasets[key] = xr.open_dataset(filename +\
                    str(self.previous_timestep.year) + '_' + str(self.previous_timestep.month).zfill(2) + file_string)
                # Rename valid_time to time if it exists
                if 'valid_time' in self.current_datasets[key].coords:
                    self.current_datasets[key] = self.current_datasets[key].rename({'valid_time': 'time'})
                if 'valid_time' in self.previous_datasets[key].coords:
                    self.previous_datasets[key] = self.previous_datasets[key].rename({'valid_time': 'time'})
                if 'pressure_level' in self.current_datasets[key].coords:
                    self.current_datasets[key] = self.current_datasets[key].rename({'pressure_level': 'level'})
                if 'pressure_level' in self.previous_datasets[key].coords:
                    self.previous_datasets[key] = self.previous_datasets[key].rename({'pressure_level': 'level'})
                self.current_datasets[key] = self.current_datasets[key].sortby('level', ascending=True)
                self.previous_datasets[key] = self.previous_datasets[key].sortby('level', ascending=True)
        return self.current_datasets, self.previous_datasets
    
    def compute_phyb_half(self, init_time, step, datasets):
        """
        Compute hybrid pressure half-levels.
        
        Parameters
        ----------
        init_time : pd.Timestamp
        step : pd.Timedelta
        datasets : dict
            Dictionary of xarray datasets
        
        Returns
        -------
        np.ndarray
            Hybrid pressure half-levels array with shape (levels, lat, lon)
        """
        
        if 'file_surface' not in datasets:
            print("Warning: Surface data file not loaded")
            return None
        
        ds_ps = datasets['file_surface']
        
        try:
            # Extract surface pressure and compute hybrid pressure half-levels
            lnsp = ds_ps.sel(time=init_time, step=step).lnsp.values
            phyb_half = (self.aa[:, np.newaxis, np.newaxis] + 
                        self.bb[:, np.newaxis, np.newaxis] * np.exp(lnsp)[np.newaxis, :, :])
            
            print(f"  Shape: {phyb_half.shape}")
            return phyb_half
            
        except KeyError as e:
            print(f"Error: Variable or time not found - {e}")
            return None
    
    def _compute_phyb_half(self):
        """
        Compute hybrid pressure half-levels for current and previous timesteps.
        Stores in self.phyb_half_current and self.phyb_half_previous.
        """
        
        self.phyb_half_current = self.compute_phyb_half(self.current_init_time, self.current_step, self.current_datasets)
        self.phyb_half_previous = self.compute_phyb_half(self.previous_init_time, self.previous_step, self.previous_datasets)
        
    def _compute_geopot(self):
        """
        Compute geopotential on model levels from temperature and specific humidity.
        """
        if self.t_current_ml is None or self.q_current_ml is None:
            print("Warning: t or q not available for geopotential calculation")
            return
        
        # Get surface geopotential from file_surface
        if 'file_surface' not in self.current_datasets:
            print("Warning: Surface data file not loaded")
            return
        
        ds_ps_current = self.current_datasets['file_surface']
        ds_ps_previous = self.previous_datasets['file_surface']
        
        try:
            # Extract surface geopotential
            surface_geopot_current = ds_ps_current.sel(time=self.current_init_time, step=self.current_step).z.values
            surface_geopot_previous = ds_ps_previous.sel(time=self.previous_init_time, step=self.previous_step).z.values
            
            # Compute geopotential on model levels
            self.geopot_current_ml = calc_geopot(
                self.t_current_ml, self.q_current_ml, self.phyb_half_current, surface_geopot_current
            )
            self.geopot_previous_ml = calc_geopot(
                self.t_previous_ml, self.q_previous_ml, self.phyb_half_previous, surface_geopot_previous
            )
            
            print(f"Computed geopotential on model levels")
            print(f"  geopot_current_ml shape: {self.geopot_current_ml.shape}")
            print(f"  geopot_previous_ml shape: {self.geopot_previous_ml.shape}")
            
        except KeyError as e:
            print(f"Error: Variable or time not found - {e}")
    
    def _extract_variables(self):
        """
        Extract u, v, t, q, w at current and previous timesteps.
        """
        if 'file_variables' not in self.current_datasets:
            print("Warning: file_variables not loaded")
            return
        
        ds_current = self.current_datasets['file_variables']
        ds_previous = self.previous_datasets['file_variables']
        
        try:
            # Extract u at current timestep
            self.u_current_ml = ds_current['u'].sel(time=self.current_init_time, step=self.current_step).values
            print(f"Loaded u on model levels at {self.current_init_time} step {self.current_step}")
            print(f"  Shape: {self.u_current_ml.shape}")
            
            # Extract u at previous timestep
            self.u_previous_ml = ds_previous['u'].sel(time=self.previous_init_time, step=self.previous_step).values
            print(f"Loaded u on model levels at {self.previous_init_time} step {self.previous_step}")
            print(f"  Shape: {self.u_previous_ml.shape}")
            
            # Extract v at current timestep
            self.v_current_ml = ds_current['v'].sel(time=self.current_init_time, step=self.current_step).values
            print(f"Loaded v on model levels at {self.current_init_time} step {self.current_step}")
            print(f"  Shape: {self.v_current_ml.shape}")
            
            # Extract v at previous timestep
            self.v_previous_ml = ds_previous['v'].sel(time=self.previous_init_time, step=self.previous_step).values
            print(f"Loaded v on model levels at {self.previous_init_time} step {self.previous_step}")
            print(f"  Shape: {self.v_previous_ml.shape}")

            # Extract t (temperature)
            self.t_current_ml = ds_current['t'].sel(time=self.current_init_time, step=self.current_step).values
            print(f"Loaded t on model levels at {self.current_init_time} step {self.current_step}")
            print(f"  Shape: {self.t_current_ml.shape}")
            
            self.t_previous_ml = ds_previous['t'].sel(time=self.previous_init_time, step=self.previous_step).values
            print(f"Loaded t on model levels at {self.previous_init_time} step {self.previous_step}")
            print(f"  Shape: {self.t_previous_ml.shape}")
            
            # Extract q (specific humidity)
            self.q_current_ml = ds_current['q'].sel(time=self.current_init_time, step=self.current_step).values
            print(f"Loaded q on model levels at {self.current_init_time} step {self.current_step}")
            print(f"  Shape: {self.q_current_ml.shape}")
            
            self.q_previous_ml = ds_previous['q'].sel(time=self.previous_init_time, step=self.previous_step).values
            print(f"Loaded q on model levels at {self.previous_init_time} step {self.previous_step}")
            print(f"  Shape: {self.q_previous_ml.shape}")
            
            # Extract w (vertical velocity)
            self.w_current_ml = ds_current['w'].sel(time=self.current_init_time, step=self.current_step).values
            print(f"Loaded w on model levels at {self.current_init_time} step {self.current_step}")
            print(f"  Shape: {self.w_current_ml.shape}")
            
            self.w_previous_ml = ds_previous['w'].sel(time=self.previous_init_time, step=self.previous_step).values
            print(f"Loaded w on model levels at {self.previous_init_time} step {self.previous_step}")
            print(f"  Shape: {self.w_previous_ml.shape}")
            
        except KeyError as e:
            print(f"Error: Variable or time not found - {e}")

    def extract_plev_variables(self):
        """
        Extract variables from pressure-level netCDF data.
        """
        if 'file_u' in self.current_datasets:
            self.u_current = self.current_datasets['file_u'].sel(time=self.timestep).u.values
            print(f"Loaded u on pressure levels at {self.timestep}")
            print(f"  Shape: {self.u_current.shape}")
        
        if 'file_u' in self.previous_datasets:
            self.u_previous = self.previous_datasets['file_u'].sel(time=self.previous_timestep).u.values
            print(f"Loaded u on pressure levels at {self.previous_timestep}")
            print(f"  Shape: {self.u_previous.shape}")
        
        if 'file_v' in self.current_datasets:
            self.v_current = self.current_datasets['file_v'].sel(time=self.timestep).v.values
            print(f"Loaded v on pressure levels at {self.timestep}")
            print(f"  Shape: {self.v_current.shape}")
        
        if 'file_v' in self.previous_datasets:
            self.v_previous = self.previous_datasets['file_v'].sel(time=self.previous_timestep).v.values
            print(f"Loaded v on pressure levels at {self.previous_timestep}")
            print(f"  Shape: {self.v_previous.shape}")
        
        if 'file_t' in self.current_datasets:
            ds = self.current_datasets['file_t']
            self.t_current = ds.sel(time=self.timestep).t.values
            print(f"Loaded t on pressure levels at {self.timestep}")
            print(f"  Shape: {self.t_current.shape}")
            # Read pressure levels
            self.plev = np.float64(ds.level.values)
            print(self.plev)
            # Compute plev_half (midpoints between levels, with 0 at the top and psurf_hPa at the bottom)
            midpoints = 0.5 * (self.plev[:-1] + self.plev[1:])
            self.plev_half = np.concatenate([[0.0], midpoints, [psurf_hPa]])
            self.pdiff = self.plev_half[1:] - self.plev_half[:-1]
            # x_spacing: Distance between longitude points (depends on latitude)
            self.latitude = np.float64(ds.latitude.values)
            self.longitude = np.float64(ds.longitude.values)
            self.x_spacing = np.deg2rad(0.25) * np.cos(np.deg2rad(self.latitude)) * r_earth
            # y_spacing: Distance between latitude points (constant)
            self.y_spacing = np.deg2rad(0.25) * r_earth

        
        if 'file_t' in self.previous_datasets:
            self.t_previous = self.previous_datasets['file_t'].sel(time=self.previous_timestep).t.values
            print(f"Loaded t on pressure levels at {self.previous_timestep}")
            print(f"  Shape: {self.t_previous.shape}")
        
        if 'file_q' in self.current_datasets:
            self.q_current = self.current_datasets['file_q'].sel(time=self.timestep).q.values
            print(f"Loaded q on pressure levels at {self.timestep}")
            print(f"  Shape: {self.q_current.shape}")
        
        if 'file_q' in self.previous_datasets:
            self.q_previous = self.previous_datasets['file_q'].sel(time=self.previous_timestep).q.values
            print(f"Loaded q on pressure levels at {self.previous_timestep}")
            print(f"  Shape: {self.q_previous.shape}")

        if 'file_w' in self.current_datasets:
            self.w_current = self.current_datasets['file_w'].sel(time=self.timestep).w.values
            print(f"Loaded w on pressure levels at {self.timestep}")
            print(f"  Shape: {self.w_current.shape}")

        if 'file_w' in self.previous_datasets:
            self.w_previous = self.previous_datasets['file_w'].sel(time=self.previous_timestep).w.values
            print(f"Loaded w on pressure levels at {self.previous_timestep}")
            print(f"  Shape: {self.w_previous.shape}")

        if 'file_z' in self.current_datasets:
            self.geopot_current = self.current_datasets['file_z'].sel(time=self.timestep).z.values
            print(f"Loaded geopotential on pressure levels at {self.timestep}")
            print(f"  Shape: {self.geopot_current.shape}")

        if 'file_z' in self.previous_datasets:
            self.geopot_previous = self.previous_datasets['file_z'].sel(time=self.previous_timestep).z.values
            print(f"Loaded geopotential on pressure levels at {self.previous_timestep}")
            print(f"  Shape: {self.geopot_previous.shape}")

    
    def _extract_param_tendencies(self):
        """
        Extract tendencies due to parametrizations.
        """
        if 'file_tendencies' not in self.current_datasets:
            print("Warning: file_tendencies not loaded")
            return
        
        ds_current = self.current_datasets['file_tendencies']
        
        try:
            # Extract temperature parametrization tendency
            self.t_param_ml = ds_current['avg_ttpm'].sel(time=self.current_init_time, step=self.current_step).values
            print(f"Loaded avg_ttpm (t_param) at {self.current_init_time} step {self.current_step}")
            print(f"  Shape: {self.t_param_ml.shape}")
            
            # Extract specific humidity parametrization tendency
            self.q_param_ml = ds_current['avg_qtpm'].sel(time=self.current_init_time, step=self.current_step).values
            print(f"Loaded avg_qtpm (q_param) at {self.current_init_time} step {self.current_step}")
            print(f"  Shape: {self.q_param_ml.shape}")
            
            # Extract shortwave radiative tendency
            self.shortwave_ml = ds_current['avg_ttswr'].sel(time=self.current_init_time, step=self.current_step).values
            print(f"Loaded avg_ttswr (shortwave) at {self.current_init_time} step {self.current_step}")
            print(f"  Shape: {self.shortwave_ml.shape}")
            
            # Extract longwave radiative tendency
            self.longwave_ml = ds_current['avg_ttlwr'].sel(time=self.current_init_time, step=self.current_step).values
            print(f"Loaded avg_ttlwr (longwave) at {self.current_init_time} step {self.current_step}")
            print(f"  Shape: {self.longwave_ml.shape}")
            
        except KeyError as e:
            print(f"Error: Variable or time not found - {e}")
    
    def _interpolate_variables_to_pressure(self):
        """
        Interpolate u, v, t, q, w from model levels to pressure levels using interpolate_to_pressure.
        """
        if self.u_current_ml is None or self.phyb_half_current is None:
            print("Warning: model level data or phyb_half not available")
            return
        
        try:
            # Get surface pressure for current and previous timesteps (last level of phyb_half)
            psurf_current = self.phyb_half_current[-1]
            psurf_previous = self.phyb_half_previous[-1]
            
            # Interpolate u to pressure levels
            self.u_current = interpolate_to_pressure(
                self.u_current_ml, psurf_current, self.plev * 100.0, self.aa, self.bb
            )
            self.u_previous = interpolate_to_pressure(
                self.u_previous_ml, psurf_previous, self.plev * 100.0, self.aa, self.bb
            )
            
            print(f"Interpolated u to pressure levels")
            print(f"  u_current shape: {self.u_current.shape}")
            print(f"  u_previous shape: {self.u_previous.shape}")
            
            # Interpolate v to pressure levels
            self.v_current = interpolate_to_pressure(
                self.v_current_ml, psurf_current, self.plev * 100.0, self.aa, self.bb
            )
            self.v_previous = interpolate_to_pressure(
                self.v_previous_ml, psurf_previous, self.plev * 100.0, self.aa, self.bb
            )
            
            print(f"Interpolated v to pressure levels")
            print(f"  v_current shape: {self.v_current.shape}")
            print(f"  v_previous shape: {self.v_previous.shape}")
            
            # Interpolate t, q, w to pressure levels if available
            if self.t_current_ml is not None and self.t_previous_ml is not None:
                self.t_current = interpolate_to_pressure(
                    self.t_current_ml, psurf_current, self.plev * 100.0, self.aa, self.bb
                )
                self.t_previous = interpolate_to_pressure(
                    self.t_previous_ml, psurf_previous, self.plev * 100.0, self.aa, self.bb
                )
                print(f"Interpolated t to pressure levels")
                print(f"  t_current shape: {self.t_current.shape}")
                print(f"  t_previous shape: {self.t_previous.shape}")
            
            if self.q_current_ml is not None and self.q_previous_ml is not None:
                self.q_current = interpolate_to_pressure(
                    self.q_current_ml, psurf_current, self.plev * 100.0, self.aa, self.bb
                )
                self.q_previous = interpolate_to_pressure(
                    self.q_previous_ml, psurf_previous, self.plev * 100.0, self.aa, self.bb
                )
                print(f"Interpolated q to pressure levels")
                print(f"  q_current shape: {self.q_current.shape}")
                print(f"  q_previous shape: {self.q_previous.shape}")
            
            if self.w_current_ml is not None and self.w_previous_ml is not None:
                self.w_current = interpolate_to_pressure(
                    self.w_current_ml, psurf_current, self.plev * 100.0, self.aa, self.bb
                )
                self.w_previous = interpolate_to_pressure(
                    self.w_previous_ml, psurf_previous, self.plev * 100.0, self.aa, self.bb
                )
                print(f"Interpolated w to pressure levels")
                print(f"  w_current shape: {self.w_current.shape}")
                print(f"  w_previous shape: {self.w_previous.shape}")
            
            # Interpolate geopotential to pressure levels if available
            if self.geopot_current_ml is not None and self.geopot_previous_ml is not None:
                self.geopot_current = interpolate_to_pressure(
                    self.geopot_current_ml, psurf_current, self.plev * 100.0, self.aa, self.bb
                )
                self.geopot_previous = interpolate_to_pressure(
                    self.geopot_previous_ml, psurf_previous, self.plev * 100.0, self.aa, self.bb
                )
                print(f"Interpolated geopotential to pressure levels")
                print(f"  geopot_current shape: {self.geopot_current.shape}")
                print(f"  geopot_previous shape: {self.geopot_previous.shape}")
            
            # Interpolate parametrization tendencies to pressure levels if available
            if self.t_param_ml is not None:
                self.t_param = interpolate_to_pressure(
                    self.t_param_ml, psurf_current, self.plev * 100.0, self.aa, self.bb
                )
                print(f"Interpolated t_param to pressure levels")
                print(f"  t_param shape: {self.t_param.shape}")
            
            if self.q_param_ml is not None:
                self.q_param = interpolate_to_pressure(
                    self.q_param_ml, psurf_current, self.plev * 100.0, self.aa, self.bb
                )
                print(f"Interpolated q_param to pressure levels")
                print(f"  q_param shape: {self.q_param.shape}")
            
            if self.shortwave_ml is not None:
                self.shortwave = interpolate_to_pressure(
                    self.shortwave_ml, psurf_current, self.plev * 100.0, self.aa, self.bb
                )
                print(f"Interpolated shortwave to pressure levels")
                print(f"  shortwave shape: {self.shortwave.shape}")
            
            if self.longwave_ml is not None:
                self.longwave = interpolate_to_pressure(
                    self.longwave_ml, psurf_current, self.plev * 100.0, self.aa, self.bb
                )
                print(f"Interpolated longwave to pressure levels")
                print(f"  longwave shape: {self.longwave.shape}")
            
        except Exception as e:
            print(f"Error interpolating to pressure levels - {e}")
    
    def _compute_mse(self):
        """
        Calculate moist static energy (MSE) at pressure levels.
        MSE = t*cp + q*lh + geopot
        """
        if self.t_current is None or self.q_current is None or self.geopot_current is None:
            print("Warning: t, q, or geopot on pressure levels not available for MSE calculation")
            return
        
        try:
            # Calculate MSE = t*cp + q*lh + geopot
            self.mse_current = self.t_current * cp + self.q_current * lh + self.geopot_current
            self.mse_previous = self.t_previous * cp + self.q_previous * lh + self.geopot_previous
            
            print(f"Computed MSE on pressure levels")
            print(f"  mse_current shape: {self.mse_current.shape}")
            print(f"  mse_previous shape: {self.mse_previous.shape}")
            
        except Exception as e:
            print(f"Error computing MSE - {e}")
    
    def _compute_mse_tendency(self):
        """
        Compute MSE tendency (rate of change) between two timesteps.
        tendency = (mse_current - mse_previous) / time_interval_in_seconds
        """
        if self.mse_current is None or self.mse_previous is None:
            print("Warning: MSE not available for tendency calculation")
            return
        
        try:
            # Convert time interval to seconds
            time_delta = pd.Timedelta(self.time_interval)
            time_interval_seconds = time_delta.total_seconds()
            
            # Compute tendency
            self.mse_tendency = (self.mse_current - self.mse_previous) / time_interval_seconds
            
            print(f"Computed MSE tendency")
            print(f"  Time interval: {self.time_interval} ({time_interval_seconds} seconds)")
            print(f"  mse_tendency shape: {self.mse_tendency.shape}")
            
        except Exception as e:
            print(f"Error computing MSE tendency - {e}")
    
    def _compute_param_mse_tendencies(self):
        """
        Compute MSE tendencies from parametrizations.
        mse_tend_sw = c_p * self.shortwave 
        mse_tend_lw = c_p * self.longwave
        mse_tend_conv = c_p * self.t_param - mse_tend_sw - mse_tend_lw + lh * self.q_param
        """
        if self.shortwave is None or self.longwave is None or self.t_param is None or self.q_param is None:
            print("Warning: Parametrization tendencies not available")
            return
        
        try:
            # Compute radiative MSE tendency
            self.mse_tend_sw = cp * self.shortwave
            self.mse_tend_lw = cp * self.longwave
            
            # Compute convective MSE tendency
            self.mse_tend_conv = cp * self.t_param - self.mse_tend_sw - self.mse_tend_lw + lh * self.q_param
            
            print(f"Computed parametrization MSE tendencies")
            print(f"  mse_tend_sw shape: {self.mse_tend_sw.shape}")
            print(f"  mse_tend_sw range: [{self.mse_tend_sw.min():.4e}, {self.mse_tend_sw.max():.4e}]")
            print(f"  mse_tend_lw shape: {self.mse_tend_lw.shape}")
            print(f"  mse_tend_lw range: [{self.mse_tend_lw.min():.4e}, {self.mse_tend_lw.max():.4e}]")
            print(f"  mse_tend_conv shape: {self.mse_tend_conv.shape}")
            print(f"  mse_tend_conv range: [{self.mse_tend_conv.min():.4e}, {self.mse_tend_conv.max():.4e}]")
            
        except Exception as e:
            print(f"Error computing parametrization MSE tendencies - {e}")
    
    def _compute_temporal_averages(self):
        """
        Compute temporal averages between current and previous timesteps.
        """
        if self.mse_current is None or self.u_current is None or self.v_current is None or self.w_current is None:
            print("Warning: Pressure-level variables not available for averaging")
            return
        
        try:
            # Average between current and previous timesteps
            self.mse = (self.mse_current + self.mse_previous) / 2.0
            self.t = (self.t_current + self.t_previous) / 2.0
            self.geopot = (self.geopot_current + self.geopot_previous) / 2.0
            self.u = (self.u_current + self.u_previous) / 2.0
            self.v = (self.v_current + self.v_previous) / 2.0
            self.w = (self.w_current + self.w_previous) / 2.0
            
            print(f"Computed temporal averages between timesteps")
            print(f"  mse shape: {self.mse.shape}")
            print(f"  mse range: [{self.mse.min():.4e}, {self.mse.max():.4e}]")
            print(f"  u shape: {self.u.shape}")
            print(f"  u range: [{self.u.min():.4e}, {self.u.max():.4e}]")
            print(f"  v shape: {self.v.shape}")
            print(f"  v range: [{self.v.min():.4e}, {self.v.max():.4e}]")
            print(f"  w shape: {self.w.shape}")
            print(f"  w range: [{self.w.min():.4e}, {self.w.max():.4e}]")
            
        except Exception as e:
            print(f"Error computing temporal averages - {e}")
    
    def _compute_mse_fluxes(self):
        """
        Compute MSE fluxes from moist static energy and wind components.
        zonal_flux = mse * u
        merid_flux = mse * v
        vertical_flux = mse * w
        """
        if self.mse is None or self.u is None or self.v is None or self.w is None:
            print("Warning: MSE or wind components not available for flux calculation")
            return
        
        try:
            # Compute MSE fluxes
            self.zonal_flux = self.mse * self.u
            self.merid_flux = self.mse * self.v
            self.vertical_flux = self.mse * self.w
            
            print(f"Computed MSE fluxes")
            print(f"  zonal_flux shape: {self.zonal_flux.shape}")
            print(f"  zonal_flux range: [{self.zonal_flux.min():.4e}, {self.zonal_flux.max():.4e}]")
            print(f"  merid_flux shape: {self.merid_flux.shape}")
            print(f"  merid_flux range: [{self.merid_flux.min():.4e}, {self.merid_flux.max():.4e}]")
            print(f"  vertical_flux shape: {self.vertical_flux.shape}")
            print(f"  vertical_flux range: [{self.vertical_flux.min():.4e}, {self.vertical_flux.max():.4e}]")
            
        except Exception as e:
            print(f"Error computing MSE fluxes - {e}")
    
    def _compute_grid_spacing(self):
        """
        Compute grid spacing in meters for latitude-longitude grid.
        x_spacing: Distance in meters between longitudinal grid points for each latitude
        y_spacing: Distance in meters between latitudinal grid points
        """
        if 'file_surface' not in self.current_datasets:
            print("Warning: Surface data file not loaded for grid spacing calculation")
            return
        
        try:
            ds_ps = self.current_datasets['file_surface']
            
            # x_spacing: Distance between longitude points (depends on latitude)
            self.x_spacing = np.deg2rad(0.25) * np.cos(np.deg2rad(self.latitude)) * r_earth
            
            # y_spacing: Distance between latitude points (constant)
            self.y_spacing = np.deg2rad(0.25) * r_earth
            
            print(f"Computed grid spacing")
            print(f"  x_spacing shape: {self.x_spacing.shape}")
            print(f"  x_spacing range: [{self.x_spacing.min():.4e}, {self.x_spacing.max():.4e}] m")
            print(f"  y_spacing: {self.y_spacing:.4e} m")
            
        except Exception as e:
            print(f"Error computing grid spacing - {e}")
    
    def compute_zonal_div(self, var):
        """
        Compute zonal derivative part of divergence using centered differences.
        
        Parameters
        ----------
        var : np.ndarray
            3D array with shape (levels, latitude, longitude)
        
        Returns
        -------
        np.ndarray
            Zonal derivative part of divergence with same shape as input
        """
        if self.x_spacing is None:
            self._compute_grid_spacing()
        
        try:
            dx_var = np.zeros_like(var)
            for i_lat in range(len(self.x_spacing)):
                dx_var[:, i_lat, :] = np.gradient(var[:, i_lat, :], self.x_spacing[i_lat], axis=1)
            
            return dx_var
            
        except Exception as e:
            print(f"Error computing zonal derivative - {e}")
            return None
    
    def compute_meridional_div(self, var):
        """
        Compute meridional part of divergence using centered differences
        
        Parameters
        ----------
        var : np.ndarray
            3D array with shape (levels, latitude, longitude)
        
        Returns
        -------
        np.ndarray
            Meridional derivative part of divergence with same shape as input
        """
        if self.y_spacing is None:
            self._compute_grid_spacing()
        
        
        try:
            latitude = self.latitude
            cos_lat = np.cos(np.deg2rad(latitude))
            
            # Factor of -1 because the latitudes are ordered in decreasing order
            dy_var = -1.0 * np.gradient(
                var * cos_lat[np.newaxis, :, np.newaxis],
                self.y_spacing,
                axis=1
            ) / cos_lat[np.newaxis, :, np.newaxis]
            
            return dy_var
            
        except Exception as e:
            print(f"Error computing meridional derivative - {e}")
            return None
    
    def compute_meridional_grad(self, var):
        """
        Compute meridional (latitude) gradient using centered differences.
        
        Parameters
        ----------
        var : np.ndarray
            3D array with shape (levels, latitude, longitude)
        
        Returns
        -------
        np.ndarray
            Meridional gradient of var with same shape as input
        """
        if self.y_spacing is None:
            self._compute_grid_spacing()
        
        try:
            # Factor of -1 because the latitudes are ordered in decreasing order
            dy_var = (-1.0) * np.gradient(var, self.y_spacing, axis=1)
            
            return dy_var
            
        except Exception as e:
            print(f"Error computing meridional gradient - {e}")
            return None
    
    def compute_vertical_derivative(self, var):
        """
        Compute vertical (pressure) derivative of a variable using centered differences.
        
        Parameters
        ----------
        var : np.ndarray
            3D array with shape (levels, latitude, longitude)
        
        Returns
        -------
        np.ndarray
            Vertical derivative of var with same shape as input
        """
        if self.plev is None:
            self._load_model_levels()
        
        try:
            # Compute gradient along vertical axis (axis=0) using pressure levels in Pa
            dp_var = np.gradient(var, self.plev * 100.0, axis=0)
            
            return dp_var
            
        except Exception as e:
            print(f"Error computing vertical derivative - {e}")
            return None
        
    def compute_vertical_integral(self, var):
        """
        Compute mass-weighted vertical (pressure) integral of a variable.
        If the input variable is in J/kg, the output will be in J/m².
        
        Parameters
        ----------
        var : np.ndarray
            3D array with shape (levels, latitude, longitude)
        
        Returns
        -------
        np.ndarray
            Vertical integral of var with shape (latitude, longitude)
        """
        if self.pdiff is None:
            self._load_model_levels()
        
        try:
            # Compute gradient along vertical axis (axis=0) using pressure levels in hPa
            var_pint = np.nansum(var * self.pdiff[:, np.newaxis, np.newaxis], axis=0) *100.0 / g_earth
            
            return var_pint
            
        except Exception as e:
            print(f"Error computing vertical integral - {e}")
            return None
        

    def compute_vertical_integral_to_p(self, var):
        """
        Compute mass-weighted vertical (pressure) integral of a variable.
        assuming it is zero at the top of the atmosphere.
        
        Parameters
        ----------
        var : np.ndarray
            3D array with shape (levels, latitude, longitude)
        
        Returns
        -------
        np.ndarray
            Vertical integral of var with same shape as input
        """
        if self.pdiff is None:
            self._load_model_levels()
        
        try:
            # Compute gradient along vertical axis (axis=0) using pressure levels in hPa
            var_pint = np.nancumsum(var * self.pdiff[:, np.newaxis, np.newaxis], axis=0) *100.0
            
            return var_pint
            
        except Exception as e:
            print(f"Error computing vertical integral - {e}")
            return None
    
    def compute_flux_divergences(self):
        """
        Compute divergences of all MSE flux components.
        
        Computes:
        - dx_zonal_flux: zonal derivative of zonal flux (u*mse)
        - dy_merid_flux: meridional derivative of meridional flux (v*mse)
        - dp_vertical_flux: vertical derivative of vertical flux (w*mse)
        
        Stores results in self.dx_zonal_flux, self.dy_merid_flux, self.dp_vertical_flux
        """
        if self.zonal_flux is None:
            self._compute_mse_fluxes()
        self.dx_zonal_flux = self.compute_zonal_div(self.zonal_flux)
        
        if self.merid_flux is None:
            self._compute_mse_fluxes()
        self.dy_merid_flux = self.compute_meridional_div(self.merid_flux)
        
        if self.vertical_flux is None:
            self._compute_mse_fluxes()
        self.dp_vertical_flux = self.compute_vertical_derivative(self.vertical_flux)
    
    def compute_zonal_advection(self, var):
        """
        Compute zonal advection of a variable.
        zonal_advection = u * dvar/dx
        
        Parameters
        ----------
        var : np.ndarray
            3D array with shape (levels, latitude, longitude)
        
        Returns
        -------
        np.ndarray
            Zonal advection with same shape as input
        """
        if self.u is None:
            print("Warning: Zonal wind (u) not available for advection calculation")
            return None
        
        try:
            dx_var = self.compute_zonal_div(var)
            zonal_adv = self.u * dx_var
            
            return zonal_adv
            
        except Exception as e:
            print(f"Error computing zonal advection - {e}")
            return None
    
    def compute_meridional_advection(self, var):
        """
        Compute meridional advection of a variable.
        meridional_advection = v * dvar/dy
        
        Parameters
        ----------
        var : np.ndarray
            3D array with shape (levels, latitude, longitude)
        
        Returns
        -------
        np.ndarray
            Meridional advection with same shape as input
        """
        if self.v is None:
            print("Warning: Meridional wind (v) not available for advection calculation")
            return None
        
        try:
            dy_var = self.compute_meridional_grad(var)
            meridional_adv = self.v * dy_var
            
            return meridional_adv
            
        except Exception as e:
            print(f"Error computing meridional advection - {e}")
            return None
    
    def compute_vertical_advection(self, var):
        """
        Compute vertical advection of a variable.
        vertical_advection = w * dvar/dp
        
        Parameters
        ----------
        var : np.ndarray
            3D array with shape (levels, latitude, longitude)
        
        Returns
        -------
        np.ndarray
            Vertical advection with same shape as input
        """
        if self.w is None:
            print("Warning: Vertical wind (w) not available for advection calculation")
            return None
        
        try:
            dp_var = self.compute_vertical_derivative(var)
            vertical_adv = self.w * dp_var
            
            return vertical_adv
            
        except Exception as e:
            print(f"Error computing vertical advection - {e}")
            return None
        
    def param_mse_tendencies_from_hourly(self):
        """
        Interpolate tendencies due to parametrizations from model-levels to pressure-levels
        """
        data_path = self.config['data']['data_path']
        first_param_timestep = self.previous_timestep + pd.Timedelta('1h')
        first_init, first_step = get_forecast_time(first_param_timestep)
        first_filetime = first_init.normalize()
        # Extract model-level tendencies
        filename = self.config['data'].get('file_tendencies', '').strip()
        ds = xr.open_dataset(data_path + '/' + str(first_filetime)[:10] \
                        +'/'+ filename + str(first_filetime.year)+str(first_filetime.month).zfill(2)+str(first_filetime.day).zfill(2),engine='cfgrib')
        t_param_ml = ds['avg_ttpm'].sel(time=first_init, step=first_step).values
        q_param_ml = ds['avg_qtpm'].sel(time=first_init, step=first_step).values
        shortwave_ml = ds['avg_ttswr'].sel(time=first_init, step=first_step).values
        longwave_ml = ds['avg_ttlwr'].sel(time=first_init, step=first_step).values
        # Read surface pressure
        ds_ps = xr.open_dataset(data_path + '/' + str(first_filetime)[:10] \
                        +'/'+ self.config['data'].get('file_surface', '').strip() + str(first_filetime.year)+str(first_filetime.month).zfill(2)+str(first_filetime.day).zfill(2),engine='cfgrib')
        p_surf = np.exp(ds_ps.sel(time=first_init, step=first_step).lnsp.values)
        t_param = interpolate_to_pressure(
            t_param_ml, p_surf, self.plev * 100.0, self.aa, self.bb)
        q_param = interpolate_to_pressure(
            q_param_ml, p_surf, self.plev * 100.0, self.aa, self.bb)
        shortwave = interpolate_to_pressure(
            shortwave_ml, p_surf, self.plev * 100.0, self.aa, self.bb)
        longwave = interpolate_to_pressure(
            longwave_ml, p_surf, self.plev * 100.0, self.aa, self.bb)
        
        filetime_before = first_filetime
        for hourly_timestep in pd.date_range(first_param_timestep, self.timestep, freq='1h')[1:]:
            init_time, step = get_forecast_time(hourly_timestep)
            file_time = init_time.normalize()
            if file_time != filetime_before: # Open new dataset
                ds.close()
                ds_ps.close()
                ds = xr.open_dataset(data_path + '/' + str(file_time)[:10] \
                        +'/'+ filename + str(file_time.year)+str(file_time.month).zfill(2)+str(file_time.day).zfill(2),engine='cfgrib')
                ds_ps = xr.open_dataset(data_path + '/' + str(file_time)[:10] \
                        +'/'+ self.config['data'].get('file_surface', '').strip() + str(file_time.year)+str(file_time.month).zfill(2)+str(file_time.day).zfill(2),engine='cfgrib')
                filetime_before = file_time
            # Read surface pressure
            p_surf = np.exp(ds_ps.sel(time=init_time, step=step).lnsp.values)
            # Extract model-level tendencies
            t_param_ml = ds['avg_ttpm'].sel(time=init_time, step=step).values
            q_param_ml = ds['avg_qtpm'].sel(time=init_time, step=step).values
            shortwave_ml = ds['avg_ttswr'].sel(time=init_time, step=step).values
            longwave_ml = ds['avg_ttlwr'].sel(time=init_time, step=step).values
            # Interpolate to pressure levels
            t_param += interpolate_to_pressure(
                t_param_ml, p_surf, self.plev * 100.0, self.aa, self.bb)
            q_param += interpolate_to_pressure(
                q_param_ml, p_surf, self.plev * 100.0, self.aa, self.bb)
            shortwave += interpolate_to_pressure(
                shortwave_ml, p_surf, self.plev * 100.0, self.aa, self.bb)
            longwave += interpolate_to_pressure(
                longwave_ml, p_surf, self.plev * 100.0, self.aa, self.bb)
        # Average over number of hourly timesteps
        n_hours = int((self.timestep - self.previous_timestep) / pd.Timedelta('1h'))
        print(n_hours)
        self.t_param = t_param / n_hours
        self.q_param = q_param / n_hours
        self.shortwave = shortwave / n_hours
        self.longwave = longwave / n_hours
        ds.close()
        ds_ps.close()
        return t_param, q_param, shortwave, longwave
    
    def to_xarray_levlat(self, var, var_name="variable"):
        """
        Convert a 2D numpy array with dimensions pressure and latitude to an xarray DataArray with coordinates.
        
        Parameters
        ----------
        var : np.ndarray
            2D array with shape (levels, latitude)
        var_name : str
            Name for the data variable
        
        Returns
        -------
        xr.DataArray
            DataArray with dimensions (plev, latitude) and proper coordinates
        """
        
        if self.plev is None:
            self._load_model_levels()
        
        try:            
            # Create xarray DataArray with proper coordinates
            da = xr.DataArray(
                var[np.newaxis,:,:],
                coords={
                    'time': [self.timestep],
                    'plev': self.plev,
                    'latitude': self.latitude
                },
                dims=['time', 'plev', 'latitude'],
                name=var_name
            )
            
            return da
            
        except Exception as e:
            print(f"Error converting to xarray - {e}")
            return None

    def to_xarray_latlon(self, var, var_name="variable"):
        """
        Convert a 2D numpy array with dimensions latitude and longitude to an xarray DataArray with coordinates.
        
        Parameters
        ----------
        var : np.ndarray
            2D array with shape (latitude, longitude)
        var_name : str
            Name for the data variable
        
        Returns
        -------
        xr.DataArray
            DataArray with dimensions (latitude, longitude) and proper coordinates
        """
        
        try:
               
            # Create xarray DataArray with proper coordinates
            da = xr.DataArray(
                var[np.newaxis,:,:],
                coords={
                    'time': [self.timestep],
                    'latitude': self.latitude,
                    'longitude': self.longitude
                },
                dims=['time', 'latitude', 'longitude'],
                name=var_name
            )
            
            return da
            
        except Exception as e:
            print(f"Error converting to xarray - {e}")
            return None
        
    def to_xarray_3d(self, var, var_name="variable"):
        """
        Convert a 3D numpy array to an xarray DataArray with coordinates.
        
        Parameters
        ----------
        var : np.ndarray
            3D array with shape (levels, latitude, longitude)
        var_name : str
            Name for the data variable
        
        Returns
        -------
        xr.DataArray
            DataArray with dimensions (plev, latitude, longitude) and proper coordinates
        """
        
        if self.plev is None:
            self._load_model_levels()
        
        try:
            ds_ps = self.current_datasets['file_surface']
            latitude = self.latitude
            longitude = self.longitude
            
            # Create xarray DataArray with proper coordinates
            da = xr.DataArray(
                var[np.newaxis,:,:,:],
                coords={
                    'time': [self.timestep],
                    'plev': self.plev,
                    'latitude': latitude,
                    'longitude': longitude
                },
                dims=['time','plev', 'latitude', 'longitude'],
                name=var_name
            )
            
            return da
            
        except Exception as e:
            print(f"Error converting to xarray - {e}")
            return None





if __name__ == "__main__":
    mse_budget_instance = mse_budget(config_file='mse_budget.ini', timestep='2005-06-02 17:00', time_interval='1 hour')

    #mse_integral = mse_budget_instance.compute_vertical_integral(mse_budget_instance.mse_tendency)






    #mse_budget_instance.compute_flux_divergences()

"""    # Right-hand side of MSE budget
    rhs_mse = -mse_budget_instance.dx_zonal_flux -\
          mse_budget_instance.dy_merid_flux -\
          mse_budget_instance.dp_vertical_flux +\
          mse_budget_instance.mse_tend_conv +\
          mse_budget_instance.mse_tend_rad

    # Right-hand side of MSE budget with advection
    rhs_mse_adv = -mse_budget_instance.compute_zonal_advection(mse_budget_instance.mse) -\
          mse_budget_instance.compute_meridional_advection(mse_budget_instance.mse) -\
          mse_budget_instance.compute_vertical_advection(mse_budget_instance.mse) +\
          mse_budget_instance.mse_tend_conv +\
          mse_budget_instance.mse_tend_rad
    
    # Right-hand side of temperature budget
    rhs_temp = -mse_budget_instance.compute_zonal_advection(mse_budget_instance.t) -\
          mse_budget_instance.compute_meridional_advection(mse_budget_instance.t) -\
          mse_budget_instance.compute_vertical_advection(mse_budget_instance.t) +\
          mse_budget_instance.t_param -\
          mse_budget_instance.compute_vertical_advection(mse_budget_instance.geopot/cp)
    
    conti = mse_budget_instance.compute_zonal_div(mse_budget_instance.u) +\
            mse_budget_instance.compute_meridional_div(mse_budget_instance.v) +\
            mse_budget_instance.compute_vertical_derivative(mse_budget_instance.w)
    
    plot = True
    # Plot vertical slices
    if plot:
        vminmax = 1
        import matplotlib.pyplot as plt
        # The continuity equation makes the residual small
        plt.imshow((rhs_mse + conti * mse_budget_instance.mse - mse_budget_instance.mse_tendency)[:,:,300],vmin=-vminmax,vmax=vminmax)
        plt.colorbar()
        plt.show()

        # Right-hand side of MSE budget
        plt.imshow(rhs_mse[:,:,300],vmin=-vminmax,vmax=vminmax)
        plt.colorbar()
        plt.show()

        # With advection form
        plt.imshow((rhs_mse_adv - mse_budget_instance.mse_tendency)[:,:,300],vmin=-vminmax,vmax=vminmax)
        plt.colorbar()
        plt.show()

        # Without continuity
        plt.imshow((rhs_mse - mse_budget_instance.mse_tendency)[:,:,300],vmin=-vminmax,vmax=vminmax)
        plt.colorbar()
        plt.show()

        # Residual of temperature budget
        vminmax = 1e-3
        plt.imshow(((mse_budget_instance.t_current - mse_budget_instance.t_previous)/3600.0 - rhs_temp)[:,:,300],vmin=-vminmax,vmax=vminmax)
        plt.colorbar()"""

#%%

