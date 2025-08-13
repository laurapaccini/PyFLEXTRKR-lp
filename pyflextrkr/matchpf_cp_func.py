import numpy as np
import os.path
import sys
import logging
import xarray as xr
from scipy.ndimage import label
from pyflextrkr.ft_utilities import subset_ds_geolimit
from pyflextrkr.ftfunctions import circular_mean, get_cloud_boundary, find_max_indices_to_roll, subset_roll_map

def matchpf_cp_singlefile(
    feature_filename,
    feature_numbers,
    merge_feature_numbers,
    split_feature_numbers,
    config,
):
    """
    Calculate precipitation statistics within cold pool features from a single pixel file.
    Also filters cold pools based on precipitation coverage.

    Args:
        feature_filename: string
            Feature file name.
        feature_numbers: numpy array
            Feature numbers within this file.
        merge_feature_numbers: numpy array
            Feature numbers for merging features in this file.
        split_feature_numbers: numpy array
            Feature numbers for splitting features in this file.
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        out_dict: dictionary
            Dictionary containing the track statistics data.
        out_dict_attrs: dictionary
            Dictionary containing the attributes of track statistics data.
        var_names_2d: list
            List of 2D variable names.
    """
    logger = logging.getLogger(__name__)

    feature_varname = config.get("feature_varname", "feature_number")
    pf_rr_thres = config["pf_rr_thres"]
    pf_coverage_thresh = config.get("pf_coverage_thresh", 0.2)  # Default 20% coverage
    # Parameters for handling perdiodic boundary condition
    pbc_direction = config.get("pbc_direction", "none")
    max_feature_frac_x = 0.95   # Max fraction of domain size for a feature in x-direction
    max_feature_frac_y = 0.95   # Max fraction of domain size for a feature in y-direction

    fillval = config["fillval"]
    fillval_f = np.nan

    # Read feature file
    if os.path.isfile(feature_filename):
        logger.info(feature_filename)

        # Load feature data
        logger.debug("Loading feature data")
        logger.debug(feature_filename)
        ds = xr.open_dataset(
            feature_filename,
            mask_and_scale=False,
            decode_times=False,
        )
        featuremap = ds[feature_varname].data.squeeze()
        rainratemap = ds["precipitation"].data.squeeze()
        
        feature_basetime = ds["base_time"].data.squeeze()
        lon = ds["longitude"].data.squeeze()
        lat = ds["latitude"].data.squeeze()
        lon_min = np.nanmin(lon)
        lon_max = np.nanmax(lon)
        lat_min = np.nanmin(lat)
        lat_max = np.nanmax(lat)
        ds.close()

        # Get dimensions of data
        ydim, xdim = np.shape(lat)

        # Number of features
        nmatchfeature = len(feature_numbers)

        if nmatchfeature > 0:
            # Define a list of 2D variables [tracks, times]
            var_names_2d = [
                "pf_area_fraction",
                "pf_filtered",
                "total_rain",
                "rainrate_mean",
                "rainrate_max",
            ]
            # Initialize arrays
            pf_area_fraction = np.full(nmatchfeature, fillval_f, dtype=float)
            pf_filtered = np.full(nmatchfeature, False, dtype=bool)
            total_rain = np.full(nmatchfeature, fillval_f, dtype=float)
            rainrate_mean = np.full(nmatchfeature, fillval_f, dtype=float)
            rainrate_max = np.full(nmatchfeature, fillval_f, dtype=float)
            rain_lon_max = np.full(nmatchfeature, fillval_f, dtype=float)
            rain_lat_max = np.full(nmatchfeature, fillval_f, dtype=float)
            basetime = np.full(nmatchfeature, fillval_f, dtype=float)

            # Loop over each matched feature number
            for imatchfeature in range(nmatchfeature):

                ittfeaturenumber = feature_numbers[imatchfeature]
                ittmergefeaturenumber = merge_feature_numbers[imatchfeature]
                ittsplitfeaturenumber = split_feature_numbers[imatchfeature]
                basetime[imatchfeature] = feature_basetime

                # Initialize matrices for feature data
                rainrate_map = np.full((ydim, xdim), np.nan, dtype=float)
                lon_map = np.full((ydim, xdim), np.nan, dtype=float)
                lat_map = np.full((ydim, xdim), np.nan, dtype=float)
                logger.debug(
                    ("rainrate_map allocation size: ", rainrate_map.shape)
                )

                # Find matching feature number
                ifeaturelocationy, ifeaturelocationx = np.array(
                    np.where(featuremap == ittfeaturenumber)
                )
                nfeaturepix = len(ifeaturelocationy)

                if nfeaturepix > 0:
                    logger.debug("Feature is present")
                    
                    # Add merge/split pixel locations if applicable
                    ifeaturelocationx, ifeaturelocationy = add_merge_split_locations(
                        featuremap,
                        ifeaturelocationx,
                        ifeaturelocationy,
                        ittmergefeaturenumber,
                        ittsplitfeaturenumber,
                        logger
                    )
                    
                    # Fill matrices with feature data
                    logger.debug("Fill map with data")
                    rainrate_map[ifeaturelocationy, ifeaturelocationx] = np.copy(
                        rainratemap[ifeaturelocationy, ifeaturelocationx]
                    )
                    lon_map[ifeaturelocationy, ifeaturelocationx] = np.copy(
                        lon[ifeaturelocationy, ifeaturelocationx]
                    )
                    lat_map[ifeaturelocationy, ifeaturelocationx] = np.copy(
                        lat[ifeaturelocationy, ifeaturelocationx]
                    )

                    # Isolate small region of feature data
                    logger.debug("Calculate precipitation statistics")

                    # Get feature boundary
                    maxx, maxy, minx, miny = get_cloud_boundary(
                        ifeaturelocationx,
                        ifeaturelocationy,
                        xdim,
                        ydim
                    )
                    
                    # Check if feature spans across domain boundary and handle if needed
                    roll_flag = False
                    if (((maxx - minx) >= xdim * max_feature_frac_x) or \
                        ((maxy - miny) >= ydim * max_feature_frac_y)) and \
                        (pbc_direction != 'none'):
                        # Handle periodic boundary conditions - similar to original code
                        sub_mask = featuremap[miny:maxy, minx:maxx] == ittfeaturenumber
                        # import pdb; pdb.set_trace()
                        shift_x_right, shift_y_top = find_max_indices_to_roll(
                            sub_mask, xdim, ydim
                        )
                        sub_rainrate = np.copy(rainrate_map[miny:maxy, minx:maxx])
                        sub_lon = np.copy(lon_map[miny:maxy, minx:maxx])
                        sub_lat = np.copy(lat_map[miny:maxy, minx:maxx])

                        #
                        # print(f"DEBUG: Feature {ittfeaturenumber}, imatchfeature {imatchfeature}")
                        # print(f"DEBUG: sub_rainrate shape: {sub_rainrate.shape}")
                        # print(f"DEBUG: sub_rainrate has NaN: {np.all(np.isnan(sub_rainrate))}")
                        # print(f"DEBUG: sub_rainrate min/max: {np.nanmin(sub_rainrate)}, {np.nanmax(sub_rainrate)}")
                        # print(f"DEBUG: shift_x_right: {shift_x_right}, shift_y_top: {shift_y_top}")
                        # print(f"DEBUG: xdim: {xdim}, ydim: {ydim}")

                        # sub_rainrate_map = subset_roll_map(
                        #     sub_rainrate, shift_x_right, shift_y_top, xdim, ydim
                        # )
                        # lon_roll = subset_roll_map(sub_lon, shift_x_right, shift_y_top, xdim, ydim)
                        # lat_roll = subset_roll_map(sub_lat, shift_x_right, shift_y_top, xdim, ydim)
                        # roll_flag = True
                        # Check if precipitation data has any non-zero values before rolling
                        if np.any(sub_rainrate > 0) and not np.all(np.isnan(sub_rainrate)):
                            # Proceed with rolling - has valid precipitation
                            sub_rainrate_map = subset_roll_map(sub_rainrate, shift_x_right, shift_y_top, xdim, ydim)
                            lon_roll = subset_roll_map(sub_lon, shift_x_right, shift_y_top, xdim, ydim)
                            lat_roll = subset_roll_map(sub_lat, shift_x_right, shift_y_top, xdim, ydim)
                            roll_flag = True
                        else:
                            # No valid precipitation - skip rolling, use original arrays
                            sub_rainrate_map = np.copy(sub_rainrate)
                            shift_x_right = 0
                            shift_y_top = 0
                            lon_roll = None
                            lat_roll = None
                            roll_flag = False
                    else:
                        # Isolate region over the feature
                        sub_rainrate_map = np.copy(rainrate_map[miny:maxy, minx:maxx])
                        shift_x_right = 0
                        shift_y_top = 0
                        lon_roll = None
                        lat_roll = None
                        roll_flag = False
                    
                    # Count total feature area and precipitation area
                    # Extract the original feature mask for this region
                    orig_feature_mask = (featuremap[miny:maxy, minx:maxx] == ittfeaturenumber)
                    
                    if roll_flag:
                        # For rolled features, roll the feature mask to match rolled precipitation data
                        sub_feature_mask = subset_roll_map(
                            orig_feature_mask.astype(float),
                            shift_x_right, shift_y_top, xdim, ydim
                        ).astype(bool)
                    else:
                        # For normal features, use the original mask
                        sub_feature_mask = orig_feature_mask
                    
                    # Ensure both arrays have the same shape by trimming to common dimensions
                    min_y = min(sub_feature_mask.shape[0], sub_rainrate_map.shape[0])
                    min_x = min(sub_feature_mask.shape[1], sub_rainrate_map.shape[1])
                    
                    sub_feature_mask = sub_feature_mask[:min_y, :min_x]
                    sub_rainrate_map = sub_rainrate_map[:min_y, :min_x]
                    
                    # Total feature area is the number of feature pixels
                    total_feature_area = np.count_nonzero(sub_feature_mask)
                    
                    # Precipitation area is feature pixels with rain > threshold
                    precip_area = np.count_nonzero(
                        sub_feature_mask & (sub_rainrate_map > pf_rr_thres)
                    )
                    
                    # Calculate precipitation coverage fraction
                    if total_feature_area > 0:
                        pf_area_fraction[imatchfeature] = precip_area / total_feature_area
                    else:
                        pf_area_fraction[imatchfeature] = 0.0
                    
                    # Calculate basic precipitation statistics
                    # Only calculate statistics for precipitation within the feature area
                    feature_rainrates = sub_rainrate_map[sub_feature_mask]
                    if len(feature_rainrates) > 0:
                        rainrate_mean[imatchfeature] = np.nanmean(feature_rainrates)
                        rainrate_max[imatchfeature] = np.nanmax(feature_rainrates)
                        # Calculate total rainfall within the feature (sum of all feature pixels)
                        total_rain[imatchfeature] = np.nansum(feature_rainrates)
                    else:
                        rainrate_mean[imatchfeature] = 0.0
                        rainrate_max[imatchfeature] = 0.0
                        total_rain[imatchfeature] = 0.0
                    
                    # Apply filtering - mark features with sufficient precipitation
                    if pf_area_fraction[imatchfeature] >= pf_coverage_thresh:
                        pf_filtered[imatchfeature] = True
                    
                    # Enhanced debugging - always log the first few features and any filtered ones
                    if imatchfeature < 10 or pf_filtered[imatchfeature]:
                        logger.info(f"Feature {ittfeaturenumber}: total_area={total_feature_area}, "
                                   f"precip_area={precip_area}, coverage={pf_area_fraction[imatchfeature]:.3f}, "
                                   f"total_rain={total_rain[imatchfeature]:.2f}, "
                                   f"mean_rain={rainrate_mean[imatchfeature]:.2f}, "
                                   f"max_rain={rainrate_max[imatchfeature]:.2f}, "
                                   f"filtered={pf_filtered[imatchfeature]}")
                        
                        # Show precipitation data statistics for debugging (only if feature_rainrates exists)
                        if len(feature_rainrates) > 0:
                            logger.info(f"  Raw precip stats: min={np.nanmin(feature_rainrates):.3f}, "
                                       f"max={np.nanmax(feature_rainrates):.3f}, "
                                       f"non-zero count={np.count_nonzero(feature_rainrates)}/{len(feature_rainrates)}")
                        else:
                            logger.info(f"  No precipitation data for this feature")
                        logger.info(f"  Threshold={pf_rr_thres}, Coverage threshold={pf_coverage_thresh}")
                    
                    # Warn if unusual values detected
                    if total_rain[imatchfeature] > 100.0:  # More than 100 mm/hr total seems high
                        logger.warning(f"Feature {ittfeaturenumber} has unusually high total rainfall: "
                                     f"{total_rain[imatchfeature]:.2f} mm/hr")
                    if rainrate_mean[imatchfeature] > 50.0:  # More than 50 mm/hr mean seems high
                        logger.warning(f"Feature {ittfeaturenumber} has unusually high mean rainfall: "
                                     f"{rainrate_mean[imatchfeature]:.2f} mm/hr")
                    
                    # Enhanced debugging - always log the first few features and any filtered ones
                    if imatchfeature < 10 or pf_filtered[imatchfeature]:
                        logger.info(f"Feature {ittfeaturenumber}: total_area={total_feature_area}, "
                                   f"precip_area={precip_area}, coverage={pf_area_fraction[imatchfeature]:.3f}, "
                                   f"total_rain={total_rain[imatchfeature]:.2f}, "
                                   f"mean_rain={rainrate_mean[imatchfeature]:.2f}, "
                                   f"max_rain={rainrate_max[imatchfeature]:.2f}, "
                                   f"filtered={pf_filtered[imatchfeature]}")
                        
                        # Show precipitation data statistics for debugging (only if feature_rainrates exists)
                        if len(feature_rainrates) > 0:
                            logger.info(f"  Raw precip stats: min={np.nanmin(feature_rainrates):.3f}, "
                                       f"max={np.nanmax(feature_rainrates):.3f}, "
                                       f"non-zero count={np.count_nonzero(feature_rainrates)}/{len(feature_rainrates)}")
                        else:
                            logger.info(f"  No precipitation data for this feature")
                        logger.info(f"  Threshold={pf_rr_thres}, Coverage threshold={pf_coverage_thresh}")
                    
                    # Warn if unusual values detected
                    if total_rain[imatchfeature] > 100.0:  # More than 100 mm/hr total seems high
                        logger.warning(f"Feature {ittfeaturenumber} has unusually high total rainfall: "
                                     f"{total_rain[imatchfeature]:.2f} mm/hr")
                    if rainrate_mean[imatchfeature] > 50.0:  # More than 50 mm/hr mean seems high
                        logger.warning(f"Feature {ittfeaturenumber} has unusually high mean rainfall: "
                                     f"{rainrate_mean[imatchfeature]:.2f} mm/hr")
                  
                    # Find location of maximum rainfall within the feature
                    if np.any(sub_feature_mask) and len(feature_rainrates) > 0 and np.any(feature_rainrates > 0):
                        # Create a masked array with only feature precipitation
                        masked_rainrate = np.full_like(sub_rainrate_map, np.nan)
                        masked_rainrate[sub_feature_mask] = sub_rainrate_map[sub_feature_mask]
                        
                        # Find max location within feature area
                        imax_y, imax_x = np.unravel_index(
                            np.nanargmax(masked_rainrate), masked_rainrate.shape
                        )
                        
                        if roll_flag and lon_roll is not None and lat_roll is not None:
                            # Handle properly for rolled arrays
                            rain_lon_max[imatchfeature] = lon_roll[imax_y, imax_x]
                            rain_lat_max[imatchfeature] = lat_roll[imax_y, imax_x]
                        else:
                            rain_lon_max[imatchfeature] = lon[imax_y + miny, imax_x + minx]
                            rain_lat_max[imatchfeature] = lat[imax_y + miny, imax_x + minx]

            # Group outputs in dictionaries
            out_dict = {
                "pf_area_fraction": pf_area_fraction,
                "pf_filtered": pf_filtered,
                "total_rain": total_rain,
                "rainrate_mean": rainrate_mean,
                "rainrate_max": rainrate_max,
                "rain_lon_max": rain_lon_max,
                "rain_lat_max": rain_lat_max,
            }
            
            out_dict_attrs = {
                "pf_area_fraction": {
                    "long_name": "Fraction of feature area covered by precipitation",
                    "units": "fraction",
                    "_FillValue": fillval_f,
                },
                "pf_filtered": {
                    "long_name": "Flag indicating if feature passes precipitation filter",
                    "units": "boolean",
                    "comment": f"True if precipitation covers at least {pf_coverage_thresh*100}% of feature area"
                },
                "total_rain": {
                    "long_name": "Total precipitation within feature",
                    "units": "mm/h",
                    "_FillValue": fillval_f,
                },
               
                "rainrate_mean": {
                    "long_name": "Mean rain rate within feature",
                    "units": "mm/h",
                    "_FillValue": fillval_f,
                },
                "rainrate_max": {
                    "long_name": "Maximum rain rate within feature",
                    "units": "mm/h",
                    "_FillValue": fillval_f,
                },
                "rain_lon_max": {
                    "long_name": "Longitude with maximum rain rate",
                    "units": "degrees",
                    "_FillValue": fillval_f,
                },
                "rain_lat_max": {
                    "long_name": "Latitude with maximum rain rate",
                    "units": "degrees",
                    "_FillValue": fillval_f,
                },
            }

            return out_dict, out_dict_attrs, var_names_2d

        else:
            logger.info("No matching feature found in file: " + feature_filename)

    else:
        logger.info("Feature file does not exist: " + feature_filename)

def add_merge_split_locations(
        featuremap,
        ifeaturelocationx,
        ifeaturelocationy,
        ittmergefeaturenumber,
        ittsplitfeaturenumber,
        logger,
):
    """
    Add pixel location indices of merge and split features to the current feature indices.

    Args:
        featuremap: numpy array
            Map of feature numbers
        ifeaturelocationx: numpy array
            X-indices of the feature
        ifeaturelocationy: numpy array
            Y-indices of the feature
        ittmergefeaturenumber: numpy array
            Feature numbers for merging features
        ittsplitfeaturenumber: numpy array
            Feature numbers for splitting features
        logger: logging object
            Logger for message output

    Returns:
        ifeaturelocationx: numpy array
            Updated X-indices including merge/split features
        ifeaturelocationy: numpy array
            Updated Y-indices including merge/split features
    """
    # Check if any features are merging
    logger.debug("Finding mergers")
    idmergefeaturenumber = np.array(np.where(ittmergefeaturenumber > 0))[0,:]
    nmergefeature = len(idmergefeaturenumber)
    if nmergefeature > 0:
        # Loop over each merging feature
        for imc in idmergefeaturenumber:
            # Find location of the merging feature
            (
                imergelocationy,
                imergelocationx,
            ) = np.array(
                np.where(featuremap == ittmergefeaturenumber[imc])
            )
            nmergepix = len(imergelocationy)

            # Add merge pixels to feature pixels
            if nmergepix > 0:
                ifeaturelocationy = np.hstack(
                    (ifeaturelocationy, imergelocationy)
                )
                ifeaturelocationx = np.hstack(
                    (ifeaturelocationx, imergelocationx)
                )
                
    # Check if any features are splitting
    logger.debug("Finding splits")
    idsplitfeaturenumber = np.array(np.where(ittsplitfeaturenumber > 0))[0,:]
    nsplitfeature = len(idsplitfeaturenumber)
    if nsplitfeature > 0:
        # Loop over each splitting feature
        for imc in idsplitfeaturenumber:
            # Find location of the splitting feature
            (
                isplitlocationy,
                isplitlocationx,
            ) = np.array(
                np.where(featuremap == ittsplitfeaturenumber[imc])
            )
            nsplitpix = len(isplitlocationy)

            # Add split pixels to feature pixels
            if nsplitpix > 0:
                ifeaturelocationy = np.hstack(
                    (ifeaturelocationy, isplitlocationy)
                )
                ifeaturelocationx = np.hstack(
                    (ifeaturelocationx, isplitlocationx)
                )
                
    return ifeaturelocationx, ifeaturelocationy