import os
import sys
import numpy as np
import time
import xarray as xr
import pandas as pd
import logging
from scipy import integrate
from scipy.ndimage import gaussian_filter
from pyflextrkr.ftfunctions import sort_renumber, skimage_watershed, adjust_pbc_watershed
from pyflextrkr.ft_utilities import get_timestamp_from_filename_single
import re

def update_time(ds):
    """
    Convert time from seconds since simulation started to proper datetime.
    """
    import pandas as pd
    
    # Reference start date
    start_date = pd.Timestamp('2000-01-01 00:00:00')
    # Convert time in seconds to pandas Timedelta
    timedeltas = pd.to_timedelta(ds.time.values, unit='s')
    # Convert time in seconds to datetime format
    pd_time = pd.Series(start_date + timedeltas) 
    ds = ds.assign_coords(time=pd_time.values)
    ds = ds.sortby('time')
    
    return ds

def get_h5_files_timerange(
    data_path,
    start_basetime,
    end_basetime,
):
    """
    Get .h5 files within specified time range by reading time coordinates from files.
    
    Args:
        data_path: string
            Data directory name.
        start_basetime: int
            Start base time (Epoch time).
        end_basetime: int
            End base time (Epoch time).
    
    Returns:
        data_filenames: list
            List of data file names with full path.
        files_basetime: numpy array
            Array of file base time.
        files_datestring: list
            List of file date string.
        files_timestring: list
            List of file time string.
    """
    import glob
    import os
    import numpy as np
    import pandas as pd
    import xarray as xr
    
    logger = logging.getLogger(__name__)
    
    # Find all .h5 files in the directory
    h5_files = sorted(glob.glob(os.path.join(data_path, "*.h5")))
    
    data_filenames = []
    files_basetime = []
    files_datestring = []
    files_timestring = []
    
    logger.info(f"Found {len(h5_files)} .h5 files, checking time coordinates...")
    
    for file_path in h5_files:
        try:
            # Try to read time coordinate from file using netcdf4 engine
            with xr.open_dataset(file_path, engine='netcdf4') as ds:
                if 'time' in ds.coords:
                    # # Debug: log original time values
                    # original_time = ds.time.values[0]
                    # logger.debug(f"File {os.path.basename(file_path)}: original time = {original_time}")
                    
                    # Apply time conversion
                    ds_time = update_time(ds)
                    time_coord = ds_time.time.values[0]  # Take first time point
                    
                    # Debug: log converted time
                    # logger.debug(f"File {os.path.basename(file_path)}: converted time = {time_coord}")
                    
                    # Convert to basetime (Epoch time)
                    if isinstance(time_coord, np.datetime64):
                        time_coord = pd.to_datetime(time_coord)
                    basetime = int(time_coord.timestamp())
                    
                    # Debug: log basetime and range check
                    # logger.debug(f"File {os.path.basename(file_path)}: basetime = {basetime}, range = [{start_basetime}, {end_basetime}]")
                    
                    # Check if within time range
                    if start_basetime <= basetime <= end_basetime:
                        data_filenames.append(file_path)
                        files_basetime.append(basetime)
                        
                        # Create date and time strings for compatibility
                        time_str = time_coord.strftime("%Y%m%d_%H%M%S")
                        files_datestring.append(time_str[:8])
                        files_timestring.append(time_str[9:])
                        
                        logger.debug(f"Added file: {os.path.basename(file_path)} - {time_coord}")
                else:
                    logger.warning(f"No time coordinate found in {os.path.basename(file_path)}")
                    
        except Exception as e:
            logger.warning(f"Could not read time from {os.path.basename(file_path)}: {e}")
            continue
    
    logger.info(f"Selected {len(data_filenames)} files within time range")
    
    return (
        data_filenames,
        np.array(files_basetime),
        files_datestring,
        files_timestring,
    )

#---------------------------------------------------------------------------------
def calculate_buoyancy_dmean(ds, var='thetav'):
    """
    Calculates buoyancy from potential virtual temperature.

    Arguments:
        ds: Xarray Dataset
            Input DataSet.
        var: string
            Name of potential virtual temperature variable.

    Returns:
        Buoyancy: Xarray DataArray
            Buoyancy field.
    """
    # Calculate spatial mean over X and Y
    spatial_mean = ds[var].mean(dim=("X", "Y"), keep_attrs=True)

    # Calculate buoyancy
    buoyancy = 9.81 * (ds[var] - spatial_mean) / spatial_mean

    # Create the buoyancy dataset
    ds_buoyancy = buoyancy.to_dataset(name='buoyancy')
    
    return ds_buoyancy

#---------------------------------------------------------------------------------
def calc_coldpool_intensity(buoy, zz, threshold=-0.005, min_cp_depth=0):
    """
    Calculates cold-pool intensity, essentially the integrated buoyancy from the surface up to where the threshold value is crossed.
    Original author: William.Gustafson@pnnl.gov
    edited by: Laura.Paccini@pnnl.gov
    Date: 30-Oct-2024

    Arguments:
        buoy: numpy array [z, y, x]
            Buoyancy 3D array.
        zz: numpy array [z, y, x]
            Height profile on center points [m]
        threshold: float, default=-0.005
            Buoyancy threshold for determining top of cold pool [m/s^2]
        min_cp_depth: float, default=0
            Minimum cold pool depth threshold [m].

    Returns:
        cp_dict: dictionary
            Dictionary containing cold pool variables.
    """
    # Find column-min buoyancy (buoy dimensions: [Z, Y, X])
    # jpool, ipool = np.where(np.nanmin(buoy, axis=0) < threshold)

    # Find lowest level buoyancy < threshold (i.e., 'surface' cold pools)
    jpool, ipool = np.where(np.squeeze(buoy[0,:,:]) < threshold)
    npool = len(jpool)
    
    # Initialize arrays
    nk, nj, ni = buoy.shape
    cp_intensity = np.zeros([nj, ni], dtype=np.float32)
    depthpool = np.zeros_like(cp_intensity)
    depth_base = np.zeros_like(cp_intensity)
    depth_top = np.zeros_like(cp_intensity)

    # Loop over each column that contains cold pool
    for m in range(npool):
        buoy_profile = buoy[:, jpool[m], ipool[m]]
        kpool = np.where(np.squeeze(buoy_profile < threshold))[0]

        gap = 1  # num. of levels of separation allowed before declaring a new layer
        layers = np.split(kpool, np.where(np.diff(kpool) > gap)[0]+1)
        # Ensure layers is non-empty and the first layer is non-empty
        if len(layers) > 0 and len(layers[0]) > 0:
            # Get layer top & bottom indices
            ktop = layers[0][-1]
            kbot = layers[0][0]
            ktopp1 = ktop + 1
            # Get height for this grid
            zz_profile = zz[:, jpool[m], ipool[m]]
            # Get cold pool depth
            z_depth = zz_profile[ktop] - zz_profile[kbot]
            # Check if it exceeds min coldpool depth
            if z_depth > min_cp_depth:
                # Vertically integrate buoyancy
                integrated_buoyancy = integrate.simpson(buoy_profile[kbot:ktopp1], x=zz_profile[kbot:ktopp1])
                # integrated_buoyancy = integrate.simps(buoy_profile[kbot:ktopp1], zz_profile[kbot:ktopp1])
                # depthpool[jpool[m], ipool[m]] = zz[ktop-kbot, jpool[m], ipool[m]]
                # depth_base[jpool[m], ipool[m]] = zz[kbot, jpool[m], ipool[m]]
                depthpool[jpool[m], ipool[m]] = z_depth
                depth_base[jpool[m], ipool[m]] = zz_profile[kbot]
                depth_top[jpool[m], ipool[m]] = zz_profile[ktop]
        
                # Compute cold pool intensity (Bryan & Parker 2010; Rotunno et al. 1988)
                with np.errstate(invalid='ignore'):
                    cp_intensity[jpool[m], ipool[m]] = np.sqrt(-2.0 * integrated_buoyancy)

                # import matplotlib.pyplot as plt
                # import pdb; pdb.set_trace()

    # Put variables to dictionary
    cp_dict = {
        'cp_depth': depthpool,
        'cp_intensity': cp_intensity,
        'cp_base': depth_base,
        'cp_top': depth_top,
    }
    return cp_dict

#---------------------------------------------------------------------------------
def filter_nonsfc_coldpool(cp_dict, zz_bot):
    """
    Filter non-surface cold pools.

    Arguments:
        cp_dict: dictionary
            Dictionary containing cold pool variables.
        zz_bot: numpy array [y, x]
            Height of the bottom level.

    Returns:
        out_dict: dictionary
            Dictionary containing cold pool variables.
    """
    # Mask of grids where cold pool base != lowest level
    mask = cp_dict['cp_base'] != zz_bot
    # Copy arrays
    cp_intensity_f = np.copy(cp_dict['cp_intensity'])
    cp_depth_f = np.copy(cp_dict['cp_depth'])
    cp_base_f = np.copy(cp_dict['cp_base'])
    cp_top_f = np.copy(cp_dict['cp_top'])
    # Filter
    cp_intensity_f[mask] = 0
    cp_depth_f[mask] = np.nan
    cp_base_f[mask] = np.nan
    cp_top_f[mask] = np.nan
    # Output dictionary
    out_dict = {
        'cp_intensity': cp_intensity_f,
        'cp_depth': cp_depth_f,
        'cp_base': cp_base_f,
        'cp_top': cp_top_f,
    }
    return out_dict

#---------------------------------------------------------------------------------
def idcoldpool(
    input_filename,
    config,
):
    """
    Identify cold pool.

    Arguments:
        input_filename: string
            Input data filename
        config: dictionary
            Dictionary containing config parameters

    Returns:
        cloudid_outfile: string
            Cloudid file name.
    """
    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)

    databasename = config.get("databasename")
    time_format = config.get("time_format")
    feature_varname = config.get("feature_varname", "feature_number")
    nfeature_varname = config.get("nfeature_varname", "nfeatures")
    featuresize_varname = config.get("featuresize_varname", "npix_feature")
    x_dimname = config.get("x_dimname")
    y_dimname = config.get("y_dimname")
    z_dimname = config.get("z_dimname", None)
    time_dimname = config.get("time_dimname")
    time_coordname = config.get("time_coordname")
    x_coordname = config.get("x_coordname")
    y_coordname = config.get("y_coordname")
    z_coordname = config.get("z_coordname")
    field_varname = config.get("field_varname")
    # field_thresh = config.get("field_thresh")
    min_size = config.get("min_size")
    linkpf = config.get('linkpf', 0)  # Option to include precipitation data
    pcp_varname = config.get('pcp_varname', None)  # Precipitation variable name
    pcp_convert_factor = config.get('pcp_convert_factor', 1)  # Convert precipitation factor
    # label_method = config.get("label_method", "ndimage.label")
    buoy_thresh = config.get("buoy_thresh")
    min_cp_depth = config.get("min_cp_depth")
    buoy_smooth_sigma = config.get("buoy_smooth_sigma", 1)
    pixel_radius = config.get("pixel_radius")
    # R_earth = config.get("R_earth")
    pass_varname = config.get("pass_varname", None)
    fillval = config["fillval"]

    # Read input data
    ds = xr.open_dataset(input_filename, mask_and_scale=False)
    # Get dimension names from the file
    dims_file = []
    for key in ds.sizes: dims_file.append(key)
    # Find extra dimensions beyond [time, z, y, x]
    dims_keep = [time_dimname, z_dimname, y_dimname, x_dimname]
    dims_drop = list(set(dims_file) - set(dims_keep))
    # Reorder Dataset dimensions
    if z_dimname is not None:
        # Drop extra dimensions, reorder to [time, z, y, x]
        ds = ds.drop_dims(dims_drop).transpose(
            time_dimname, z_dimname, y_dimname, x_dimname, missing_dims='ignore',
        )
    else:
        # Drop extra dimensions, reorder to [time, y, x]
        ds = ds.drop_dims(dims_drop).transpose(
            time_dimname, y_dimname, x_dimname, missing_dims='ignore',
        )

    # Check if time dimension exists in the DataSet
    if time_dimname not in ds.sizes:
        # Add a 'time' dimension with size 1 to all variables
        ds = ds.expand_dims(time_dimname, axis=0)

    # Check if time coordinate exists in the DataSet
    if time_coordname not in ds:
        # Handle no time coordinate in Dataset
        logger.warning(f'No time coordinate: {time_coordname} found in input data')
        logger.warning(f'Will estimate time from filename based on time_format in config: {time_format}')
        # Get Timestamp from filename
        file_timestamp = get_timestamp_from_filename_single(
            input_filename, databasename, time_format=time_format,
        )
        # Add Timestamp coordinate to the Dataset
        ds = ds.assign_coords({time_coordname:file_timestamp})
        # Add time dimension to all variables in the Dataset
        ds = xr.concat([ds], dim=time_dimname)
        logger.debug(f'Added Timestamp: {file_timestamp} calculated from filename to the input data')

    # Read data variables
    ntimes = ds.sizes[time_dimname]
    x_coord = ds.coords[x_coordname]
    y_coord = ds.coords[y_coordname]
    z_coord = ds.coords[z_coordname]
    time_decode = ds[time_coordname]
    field_var = ds[field_varname]


    has_precip = False
    # Read precipitation data if linkpf is enabled
    if linkpf == 1 and pcp_varname is not None:
        if pcp_varname in ds:
            has_precip = True
            # Convert precipitation factor to unit [mm/hour]
            precip_data = ds[pcp_varname].data * pcp_convert_factor
            logger.info(f"Using precipitation from main input file")
        else:
            # Try to load from separate file
            pcp_data_path = config.get('pcp_data_path', None)
            pcp_filebase = config.get('pcp_filebase', None)
            pcp_time_format = config.get('pcp_time_format', time_format)

            if pcp_data_path and pcp_filebase:
                # Get current timestamp from the buoyancy file
                current_timestamp = time_decode[0].values
                
                # Format timestamp for matching precipitation file
                if isinstance(current_timestamp, np.datetime64):
                    current_timestamp = pd.to_datetime(current_timestamp)
                
                timestamp_str = current_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
                
                # Build pattern for precipitation file
                pcp_file = f"{pcp_data_path}{pcp_filebase}{timestamp_str}.nc"
                
                if os.path.exists(pcp_file):
                    logger.info(f"Loading precipitation from separate file: {pcp_file}")
                    try:
                        ds_pcp = xr.open_dataset(pcp_file)
                        if pcp_varname in ds_pcp:
                            # Check dimension order and transpose if needed
                            pcp_var = ds_pcp[pcp_varname]
                            if pcp_var.dims[-2:] == ('X', 'Y'):
                                # Transpose to match expected (time, Y, X) order
                                precip_data = pcp_var.values.transpose(0, 2, 1) * pcp_convert_factor
                                logger.info("Transposed precipitation data from (time, X, Y) to (time, Y, X)")
                            else:
                                # Already in correct order
                                precip_data = pcp_var.values * pcp_convert_factor
                            has_precip = True
                            logger.info(f"Successfully loaded precipitation data from variable '{pcp_varname}'")
                            logger.info(f"Precipitation data stats: min={np.min(precip_data):.6f}, max={np.max(precip_data):.6f}, mean={np.mean(precip_data):.6f}")
                        else:
                            available_vars = list(ds_pcp.variables.keys())
                            logger.error(f"Variable '{pcp_varname}' not found in precipitation file")
                            logger.error(f"Available variables in precipitation file: {available_vars}")
                            logger.error("This will result in zero precipitation being used for all features!")
                        ds_pcp.close()
                    except Exception as e:
                        logger.error(f"Error loading precipitation file: {e}")
                else:
                    logger.warning(f"No matching precipitation file found: {pcp_file}")
            else:
                logger.warning("Precipitation path or filebase not specified")
            
    # else:
    #     has_precip = False
    
    


    ds.close()

    # Check x, y coordinate dimensions
    if (y_coord.ndim == 1) | (x_coord.ndim == 1):
        # Mesh 1D coordinate into 2D
        lon2d, lat2d = np.meshgrid(x_coord, y_coord)
        lon2d = lon2d.astype(np.float32)
        lat2d = lat2d.astype(np.float32)
    elif (y_coord.ndim == 2) | (x_coord.ndim == 2):
        lon2d = x_coord.data
        lat2d = y_coord.data

    # Check z coordinate dimensions
    if (z_coord.ndim == 1):
        # Expand z_coord to shape (Z, Y, X)
        _z_coord = z_coord.data[:, np.newaxis, np.newaxis]
        # Broadcast z_coord_expanded to match the shape of field_var
        var_shape = field_var.isel({time_dimname:0}).squeeze().shape
        zz = np.broadcast_to(_z_coord, var_shape)
    elif (z_coord.ndim == 3):
        zz = z_coord.data

    # Calculate mean lat/lon grid distance (assuming fix grid size)
    # dlon = np.mean(np.abs(np.diff(lon2d, axis=1)))
    # dlat = np.mean(np.abs(np.diff(lat2d, axis=0)))

    # Calculate grid cell area (simple cosine adjustment)
    # grid_area = (R_earth**2) * np.cos(np.deg2rad(lat2d)) * np.deg2rad(dlat) * np.deg2rad(dlon)
    grid_area = pixel_radius**2

    if pass_varname is not None:
        # Find the common variable names between the dataset and the list
        pass_varname = set(ds.data_vars) & set(pass_varname)
        # Subset the input dataset
        ds_pass = ds[pass_varname]


    # Loop over each time
    for tt in range(0, ntimes):
        # Get data at this time
        iTime = time_decode[tt]
        fvar = field_var.data[tt,:,:,:].squeeze()

        # Calculate cold pool intensity
        cp_dict = calc_coldpool_intensity(fvar, zz, threshold=buoy_thresh, min_cp_depth=min_cp_depth)

        # Filter points where cold pool bottom is not at the lowest height
        zz_bot = zz[0,:,:].squeeze()
        cp_dict = filter_nonsfc_coldpool(cp_dict, zz_bot)

        # Smooth buoyancy intensity
        cp_intensity_s = gaussian_filter(cp_dict['cp_intensity'], sigma=buoy_smooth_sigma)
        # import matplotlib.pyplot as plt
        # import pdb; pdb.set_trace()

        # Add precipitation to output if available
        if has_precip:
            # For 3D precipitation data, take the surface level
            if precip_data.ndim == 3:  # [time, y, x]
                pcp = precip_data[tt, :, :]
            else:
                pcp = precip_data
        else:
            # Create empty precipitation field if not available
            pcp = np.zeros_like(cp_dict['cp_intensity'])
            logger.warning(f"No precipitation data available - using zeros for all precipitation values")
            logger.warning(f"This will cause all cold pools to have zero precipitation statistics")
            
        ### Label feauture considering boundary conditions
        if config["pbc_direction"] != "none":

            var_number, param_dict = adjust_pbc_watershed(cp_intensity_s, config)
        else:
            # Label feature
            var_number, param_dict = skimage_watershed(cp_intensity_s, config)

        # Sort and renumber features, filter features < min_size or grid_area
        feature_mask, npix_feature = sort_renumber(var_number, min_size)

        # Get number of features
        nfeatures = np.nanmax(feature_mask)

        # Convert to basetime (i.e., Epoch time)
        iTimestamp = pd.to_datetime(iTime.dt.strftime("%Y-%m-%dT%H:%M:%S").item())
        file_basetime = np.array([(iTimestamp - pd.Timestamp('1970-01-01T00:00:00')).total_seconds()])
        # Convert to strings
        file_datestring = iTime.dt.strftime("%Y%m%d").item()
        file_timestring = iTime.dt.strftime("%H%M%S").item()
        cloudid_outfile = (
            config["tracking_outpath"] +
            config["cloudid_filebase"] +
            file_datestring +
            "_" +
            file_timestring +
            ".nc"
        )

        # Put time and nfeatures in a numpy array so that they can be set with a time dimension
        out_basetime = np.zeros(1, dtype=float)
        out_basetime[0] = file_basetime

        out_nfeatures = np.zeros(1, dtype=int)
        out_nfeatures[0] = nfeatures

        #######################################################
        # Output netcdf file
        # Define 3 variables required for tracking
        bt_attrs = {
            "long_name": "Base time in Epoch",
            "units": "Seconds since 1970-1-1 0:00:00 0:00",
        }
        featuremask_attrs = {
            "long_name": "Labeled feature number for tracking",
            "units": "unitless",
        }
        nfeatures_attrs = {
            "long_name": "Number of features labeled",
            "units": "unitless",
        }
        npix_feature_attrs = {
            "long_name": "Number of pixels for each feature",
            "units": "unitless",
        }
        # Additional variables
        cp_intensity_attrs = {
            "long_name": "Cold pool intensity (vertically integrated buoyancy)",
            "units": "m/s",
        }
        cp_depth_attrs = {
            "long_name": "Cold pool depth",
            "units": "m",
        }
        cp_base_attrs = {
            "long_name": "Cold pool base height",
            "units": "m",
        }
        cp_top_attrs = {
            "long_name": "Cold pool top height",
            "units": "m",
        }
        precip_attrs = {
            "long_name": "Precipitation rate",
            "units": "mm/hour",
            "description": f"From {pcp_varname} * {pcp_convert_factor}",
        }

        # Define variable dictionary
        var2d_dims = ["time", "lat", "lon"]
        var_dict = {
            "base_time": (["time"], out_basetime, bt_attrs),
            "longitude": (["lat", "lon"], lon2d, x_coord.attrs),
            "latitude": (["lat", "lon"], lat2d, y_coord.attrs),
            "cp_intensity": (var2d_dims, np.expand_dims(cp_dict['cp_intensity'], 0), cp_intensity_attrs),
            "cp_depth": (var2d_dims, np.expand_dims(cp_dict['cp_depth'], 0), cp_depth_attrs),
            "cp_base": (var2d_dims, np.expand_dims(cp_dict['cp_base'], 0), cp_base_attrs),
            "cp_top": (var2d_dims, np.expand_dims(cp_dict['cp_top'], 0), cp_top_attrs),
            "precipitation": (var2d_dims, np.expand_dims(pcp, 0), precip_attrs),
            feature_varname: (var2d_dims, np.expand_dims(feature_mask, 0), featuremask_attrs),
            nfeature_varname: (["time"], out_nfeatures, nfeatures_attrs),
            featuresize_varname: (["features"], npix_feature, npix_feature_attrs),
        }
        coord_dict = {
            "time": (["time"], out_basetime, bt_attrs),
            "lat": (["lat"], y_coord.data, y_coord.attrs),
            "lon": (["lon"], x_coord.data, x_coord.attrs),
            "features": (["features"], np.arange(1, nfeatures + 1)),
        }
        gattr_dict = {
            "Title": f"FeatureID file from {file_datestring}.{file_timestring}",
            "Institution": "Pacific Northwest National Laboratory",
            "Contact": "Zhe Feng: zhe.feng@pnnl.gov",
            "Created_on": time.ctime(time.time()),
            "min_size": min_size,
            # Watershed segmentation parameters
            "plm_min_distance": config.get("plm_min_distance"),
            "plm_threshold_abs": config.get("plm_threshold_abs"), 
            "plm_exclude_border": config.get("plm_exclude_border"),
            "cont_thresh": config.get("cont_thresh"),
            "buoy_smooth_sigma": config.get("buoy_smooth_sigma"),
            "area_thresh": config.get("area_thresh"),
            "label_method": config.get("label_method"),
            "buoy_thresh": config.get("buoy_thresh"),
            "min_cp_depth": config.get("min_cp_depth"),
        }
        # Add each parameter to global attribute dictionary
        for key in param_dict:
            gattr_dict[key] = param_dict[key]

        # Add pass out variables to the output variable dictionary
        if pass_varname is not None:
            # Subset the time from the pass out Dataset
            dsp = ds_pass.isel({time_coordname:tt})
            # Loop over each pass out variable list
            for ivar in pass_varname:
                var_dict[ivar] = (var2d_dims, np.expand_dims(dsp[ivar].data, 0), dsp[ivar].attrs)

        # Define xarray dataset
        dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

        # Delete file if it already exists
        if os.path.isfile(cloudid_outfile):
            os.remove(cloudid_outfile)

        # Set encoding/compression for all variables
        comp = dict(zlib=True)
        encoding = {var: comp for var in dsout.data_vars}
        # Write to netcdf file
        dsout.to_netcdf(
            path=cloudid_outfile,
            mode='w',
            format='NETCDF4',
            encoding=encoding
        )
        logger.info(f"{cloudid_outfile}")
        # import matplotlib.pyplot as plt
        # import pdb; pdb.set_trace()

    return cloudid_outfile

# When using cold pool variables pre-computed in 2D, use this function:

def idcoldpool_2d(
    input_filename,
    config,
):
    """
    Identify cold pool from 2D pre-computed cold pool variables.

    Arguments:
        input_filename: string
            Input data filename containing cp_intensity, cp_depth, cp_base
        config: dictionary
            Dictionary containing config parameters

    Returns:
        cloudid_outfile: string
            Cloudid file name.
    """
    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)

    databasename = config.get("databasename")
    time_format = config.get("time_format")
    feature_varname = config.get("feature_varname", "feature_number")
    nfeature_varname = config.get("nfeature_varname", "nfeatures")
    featuresize_varname = config.get("featuresize_varname", "npix_feature")
    x_dimname = config.get("x_dimname")
    y_dimname = config.get("y_dimname")
    time_dimname = config.get("time_dimname")
    time_coordname = config.get("time_coordname")
    x_coordname = config.get("x_coordname")
    y_coordname = config.get("y_coordname")
    
    # Cold pool variable names in input file
    cp_intensity_varname = config.get("cp_intensity_varname", "cp_intensity")
    cp_depth_varname = config.get("cp_depth_varname", "cp_depth")
    cp_base_varname = config.get("cp_base_varname", "cp_base")
    cp_top_varname = config.get("cp_top_varname", "cp_top")
    
    min_size = config.get("min_size")
    linkpf = config.get('linkpf', 0)
    pcp_varname = config.get('pcp_varname', None)
    pcp_convert_factor = config.get('pcp_convert_factor', 1)
    buoy_smooth_sigma = config.get("buoy_smooth_sigma", 1)
    pixel_radius = config.get("pixel_radius")
    pass_varname = config.get("pass_varname", None)
    fillval = config["fillval"]
    
    # Thresholds for 2D cold pool filtering
    min_cp_intensity = config.get("min_cp_intensity", 0.0)
    min_cp_depth = config.get("min_cp_depth", 0.0)

    # Read input data
    ds = xr.open_dataset(input_filename, mask_and_scale=False, engine='netcdf4')
    
    # Convert time coordinates if it's an .h5 file with simulation time
    if input_filename.endswith('.h5') and 'time' in ds.coords:
        logger.info("Converting simulation time to datetime format")
        ds = update_time(ds)
    
    # Get dimension names and clean up
    dims_file = list(ds.sizes.keys())
    dims_keep = [time_dimname, y_dimname, x_dimname]
    dims_drop = list(set(dims_file) - set(dims_keep))
    
    # Drop extra dimensions, reorder to [time, y, x]
    ds = ds.drop_dims(dims_drop).transpose(
        time_dimname, y_dimname, x_dimname, missing_dims='ignore',
    )

    # Handle time dimension and coordinate
    if time_dimname not in ds.sizes:
        ds = ds.expand_dims(time_dimname, axis=0)

    if time_coordname not in ds:
        logger.warning(f'No time coordinate: {time_coordname} found in input data')
        # This shouldn't happen for .h5 files with proper time coordinates
        logger.warning("This case should not occur with proper .h5 files")
        file_timestamp = get_timestamp_from_filename_single(
            input_filename, databasename, time_format=time_format,
        )
        ds = ds.assign_coords({time_coordname: file_timestamp})
        ds = xr.concat([ds], dim=time_dimname)
        logger.debug(f'Added Timestamp: {file_timestamp} calculated from filename')

    # Read coordinates and data
    ntimes = ds.sizes[time_dimname]
    x_coord = ds.coords[x_coordname]
    y_coord = ds.coords[y_coordname]
    time_decode = ds[time_coordname]
    
    # Read cold pool variables
    cp_intensity = ds[cp_intensity_varname]
    cp_depth = ds[cp_depth_varname] 
    cp_base = ds[cp_base_varname]
    
    if cp_top_varname in ds:
        cp_top = ds[cp_top_varname]
    else:
        cp_top = cp_base + cp_depth
        logger.info("cp_top not found, calculating from cp_base + cp_depth")

    # Handle precipitation data (similar to 3D version)
    has_precip = False
    if linkpf == 1 and pcp_varname is not None:
        if pcp_varname in ds:
            has_precip = True
            precip_data = ds[pcp_varname].data * pcp_convert_factor
            logger.info("Using precipitation from main input file")
        else:
            # Load from separate file (same logic as 3D version)
            pcp_data_path = config.get('pcp_data_path', None)
            pcp_filebase = config.get('pcp_filebase', None)
            
            if pcp_data_path and pcp_filebase:
                current_timestamp = time_decode[0].values
                if isinstance(current_timestamp, np.datetime64):
                    current_timestamp = pd.to_datetime(current_timestamp)
                
                timestamp_str = current_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
                pcp_file = f"{pcp_data_path}{pcp_filebase}{timestamp_str}.nc"
                
                if os.path.exists(pcp_file):
                    logger.info(f"Loading precipitation from: {pcp_file}")
                    try:
                        ds_pcp = xr.open_dataset(pcp_file)
                        if pcp_varname in ds_pcp:
                            # Check dimension order and transpose if needed
                            pcp_var = ds_pcp[pcp_varname]
                            if pcp_var.dims[-2:] == ('X', 'Y'):
                                # Transpose to match expected (time, Y, X) order
                                precip_data = pcp_var.values.transpose(0, 2, 1) * pcp_convert_factor
                                logger.info("Transposed precipitation data from (time, X, Y) to (time, Y, X)")
                            else:
                                # Already in correct order
                                precip_data = pcp_var.values * pcp_convert_factor
                            has_precip = True
                        ds_pcp.close()
                    except Exception as e:
                        logger.error(f"Error loading precipitation: {e}")

    ds.close()

    # Handle coordinates
    if (y_coord.ndim == 1) | (x_coord.ndim == 1):
        lon2d, lat2d = np.meshgrid(x_coord, y_coord)
        lon2d = lon2d.astype(np.float32)
        lat2d = lat2d.astype(np.float32)
    else:
        lon2d = x_coord.data
        lat2d = y_coord.data

    if pass_varname is not None:
        pass_varname = set(ds.data_vars) & set(pass_varname)
        ds_pass = ds[pass_varname]

    # Process each time step
    for tt in range(0, ntimes):
        iTime = time_decode[tt]
        
        # Get cold pool variables for this time
        cp_intensity_2d = cp_intensity.data[tt, :, :].squeeze()
        cp_depth_2d = cp_depth.data[tt, :, :].squeeze()
        cp_base_2d = cp_base.data[tt, :, :].squeeze()
        cp_top_2d = cp_top.data[tt, :, :].squeeze()
        
        # Apply filtering thresholds
        valid_mask = (cp_intensity_2d >= min_cp_intensity) & (cp_depth_2d >= min_cp_depth)
        cp_intensity_filtered = np.where(valid_mask, cp_intensity_2d, 0.0)
        
        # Smooth for watershed segmentation
        cp_intensity_s = gaussian_filter(cp_intensity_filtered, sigma=buoy_smooth_sigma)

        # Handle precipitation
        if has_precip:
            if precip_data.ndim == 3:
                pcp = precip_data[tt, :, :]
            else:
                pcp = precip_data
        else:
            pcp = np.zeros_like(cp_intensity_2d)

        # Feature labeling (same as 3D version)
        if config["pbc_direction"] != "none":
            var_number, param_dict = adjust_pbc_watershed(cp_intensity_s, config)
        else:
            var_number, param_dict = skimage_watershed(cp_intensity_s, config)

        feature_mask, npix_feature = sort_renumber(var_number, min_size)
        nfeatures = np.nanmax(feature_mask)

        # Create output file (same logic as 3D version)
        iTimestamp = pd.to_datetime(iTime.dt.strftime("%Y-%m-%dT%H:%M:%S").item())
        file_basetime = np.array([(iTimestamp - pd.Timestamp('1970-01-01T00:00:00')).total_seconds()])
        file_datestring = iTime.dt.strftime("%Y%m%d").item()
        file_timestring = iTime.dt.strftime("%H%M%S").item()
        cloudid_outfile = (
            config["tracking_outpath"] +
            config["cloudid_filebase"] +
            file_datestring +
            "_" +
            file_timestring +
            ".nc"
        )

        out_basetime = np.zeros(1, dtype=float)
        out_basetime[0] = file_basetime
        out_nfeatures = np.zeros(1, dtype=int)
        out_nfeatures[0] = nfeatures

        # Define output variables (same structure as 3D version)
        bt_attrs = {"long_name": "Base time in Epoch", "units": "Seconds since 1970-1-1 0:00:00 0:00"}
        featuremask_attrs = {"long_name": "Labeled feature number for tracking", "units": "unitless"}
        nfeatures_attrs = {"long_name": "Number of features labeled", "units": "unitless"}
        npix_feature_attrs = {"long_name": "Number of pixels for each feature", "units": "unitless"}
        cp_intensity_attrs = {"long_name": "Cold pool intensity", "units": "m/s"}
        cp_depth_attrs = {"long_name": "Cold pool depth", "units": "m"}
        cp_base_attrs = {"long_name": "Cold pool base height", "units": "m"}
        cp_top_attrs = {"long_name": "Cold pool top height", "units": "m"}
        precip_attrs = {"long_name": "Precipitation rate", "units": "mm/hour"}

        var2d_dims = ["time", "lat", "lon"]
        var_dict = {
            "base_time": (["time"], out_basetime, bt_attrs),
            "longitude": (["lat", "lon"], lon2d, x_coord.attrs),
            "latitude": (["lat", "lon"], lat2d, y_coord.attrs),
            "cp_intensity": (var2d_dims, np.expand_dims(cp_intensity_2d, 0), cp_intensity_attrs),
            "cp_depth": (var2d_dims, np.expand_dims(cp_depth_2d, 0), cp_depth_attrs),
            "cp_base": (var2d_dims, np.expand_dims(cp_base_2d, 0), cp_base_attrs),
            "cp_top": (var2d_dims, np.expand_dims(cp_top_2d, 0), cp_top_attrs),
            "precipitation": (var2d_dims, np.expand_dims(pcp, 0), precip_attrs),
            feature_varname: (var2d_dims, np.expand_dims(feature_mask, 0), featuremask_attrs),
            nfeature_varname: (["time"], out_nfeatures, nfeatures_attrs),
            featuresize_varname: (["features"], npix_feature, npix_feature_attrs),
        }
        coord_dict = {
            "time": (["time"], out_basetime, bt_attrs),
            "lat": (["lat"], y_coord.data, y_coord.attrs),
            "lon": (["lon"], x_coord.data, x_coord.attrs),
            "features": (["features"], np.arange(1, nfeatures + 1)),
        }
        gattr_dict = {
            "Title": f"FeatureID file from {file_datestring}.{file_timestring}",
            "Institution": "Pacific Northwest National Laboratory", 
            "Contact": "Laura Paccini: laurapaccini@gmail.com",
            "Created_on": time.ctime(time.time()),
            "min_size": min_size,
            "min_cp_intensity": min_cp_intensity,
            "min_cp_depth": min_cp_depth,
            "input_data_type": "2d",
            # Watershed segmentation parameters
            "plm_min_distance": config.get("plm_min_distance"),
            "plm_threshold_abs": config.get("plm_threshold_abs"), 
            "plm_exclude_border": config.get("plm_exclude_border"),
            "cont_thresh": config.get("cont_thresh"),
            "buoy_smooth_sigma": config.get("buoy_smooth_sigma"),
            "area_thresh": config.get("area_thresh"),
            "label_method": config.get("label_method"),
        }
        for key in param_dict:
            gattr_dict[key] = param_dict[key]

        # Add pass-through variables
        if pass_varname is not None:
            dsp = ds_pass.isel({time_coordname: tt})
            for ivar in pass_varname:
                var_dict[ivar] = (var2d_dims, np.expand_dims(dsp[ivar].data, 0), dsp[ivar].attrs)

        # Save output
        dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)
        
        if os.path.isfile(cloudid_outfile):
            os.remove(cloudid_outfile)

        comp = dict(zlib=True)
        encoding = {var: comp for var in dsout.data_vars}
        dsout.to_netcdf(path=cloudid_outfile, mode='w', format='NETCDF4', encoding=encoding)
        logger.info(f"{cloudid_outfile}")

    return cloudid_outfile