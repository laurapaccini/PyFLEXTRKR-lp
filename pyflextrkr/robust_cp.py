import numpy as np
import xarray as xr
import os
import sys
import time
import warnings
import logging

def define_robust_cp(config):
    """
    Identify robust cold pools based on precipitation statistics.
    Simplified version for cold pool tracking.

    Args:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        statistics_outfile: string
            Robust cold pool track statistics file name.
    """

    pfstats_filebase = config["pfstats_filebase"]
    cprobust_filebase = config.get("cprobust_filebase", "cp_tracks_robust_")
    stats_path = config["stats_outpath"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    cp_pf_min_area_thresh = config.get("cp_pf_min_area_thresh", 0.0)  # Minimum precipitation area threshold
    cp_pf_duration_thresh = config.get("cp_pf_duration_thresh", 0.5)  # Minimum duration threshold [hours]
    tracks_dimname = config["tracks_dimname"]
    times_dimname = config["times_dimname"]
    pixel_radius = config["pixel_radius"]

    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)
    logger.info("Identifying robust cold pools based on precipitation statistics")

    # Output stats file name
    statistics_outfile = f"{stats_path}{cprobust_filebase}{startdate}_{enddate}.nc"

    ######################################################
    # Load cold pool precipitation track stats
    cpstats_file = f"{stats_path}{pfstats_filebase}{startdate}_{enddate}.nc"
    logger.debug(f"cpstats_file: {cpstats_file}")

    ds_cp = xr.open_dataset(cpstats_file,
                            mask_and_scale=False,
                            decode_times=False,)
    ntracks = ds_cp.sizes[tracks_dimname]
    ntimes = ds_cp.sizes[times_dimname]

    track_duration = ds_cp["track_duration"].data
    pf_area_fraction = ds_cp["pf_area_fraction"].data
    pf_filtered = ds_cp["pf_filtered"].data
    total_rain = ds_cp["total_rain"].data
    time_res = float(ds_cp.attrs["time_resolution_hour"])
    fillval = ds_cp["track_status"].attrs["_FillValue"]

    ##################################################
    # Initialize matrices
    trackid_robustcp = []
    trackid_noncp = []

    cp_status = np.full((ntracks, ntimes), fillval, dtype=int)

    ###################################################
    # Loop through each track
    for nt in range(0, ntracks):
        logger.debug(f"Track # {nt}")

        ############################################
        # Isolate data from this track
        ilength = np.copy(track_duration[nt]).astype(int)

        # Get precipitation data for this track
        icp_pf_area_fraction = np.copy(pf_area_fraction[nt, 0:ilength])
        icp_pf_filtered = np.copy(pf_filtered[nt, 0:ilength])
        icp_total_rain = np.copy(total_rain[nt, 0:ilength])

        ######################################################
        # Apply precipitation area criteria
        # Count times when precipitation area fraction > minimum threshold
        ipf_valid = np.array(np.where(icp_pf_area_fraction > cp_pf_min_area_thresh)[0])
        nipf_valid = len(ipf_valid)

        if nipf_valid > 0:
            # Apply duration threshold to entire time period
            valid_duration = nipf_valid * time_res
            
            if valid_duration >= cp_pf_duration_thresh:
                logger.debug("Robust Cold Pool")
                
                # Find continuous periods where precipitation criteria is met
                # For simplicity, we'll mark all times where precipitation exists as robust CP
                valid_times = np.where(icp_pf_area_fraction > cp_pf_min_area_thresh)[0]
                
                if len(valid_times) > 0:
                    # Label these periods as robust cold pools
                    cp_status[nt, valid_times] = 1
                    trackid_robustcp.append(nt)
                else:
                    trackid_noncp.append(nt)
            else:
                trackid_noncp.append(nt)
                logger.debug("Not robust CP - insufficient duration")
        else:
            trackid_noncp.append(nt)
            logger.debug("Not robust CP - no valid precipitation")

    # Find track indices that are robust cold pools
    TEMP_cpstatus = np.copy(cp_status).astype(float)
    TEMP_cpstatus[TEMP_cpstatus == fillval] = np.nan
    trackid_robustcp = np.where(np.nansum(TEMP_cpstatus, axis=1) > 0)[0]
    nrobustcp = len(trackid_robustcp)

    # Stop code if no robust cold pools present
    if nrobustcp == 0:
        logger.warning("No robust cold pools found!")
        # Create an empty output file
        trackid_robustcp = np.array([])
        nrobustcp = 0
    else:
        logger.info(f"Number of robust cold pools: {int(nrobustcp)}")

    if nrobustcp > 0:
        # Isolate data associated with robust cold pools
        track_duration = track_duration[trackid_robustcp]
        cp_status = cp_status[trackid_robustcp, :]
        pf_area_fraction = pf_area_fraction[trackid_robustcp, :]
        total_rain = total_rain[trackid_robustcp, :]

        # Calculate lifetime when precipitation is present
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Count times when precipitation area > 0
            pf_lifetime_mask = pf_area_fraction > 0
            pf_lifetime = np.nansum(pf_lifetime_mask, axis=1) * time_res

        # Subset robust cold pool tracks from precipitation dataset
        dsout = ds_cp.sel(tracks=trackid_robustcp)
        # Replace tracks index
        tracks_coord = np.arange(0, nrobustcp)
        times_coord = ds_cp[times_dimname]
        dsout[tracks_dimname] = tracks_coord

        # Convert new variables to DataArrays
        pf_lifetime = xr.DataArray(
            pf_lifetime,
            coords={tracks_dimname: tracks_coord},
            dims=(tracks_dimname),
            attrs={
                "long_name": "Cold pool lifetime when precipitation is present",
                "units": "hour",
            }
        )
        cp_status = xr.DataArray(
            cp_status,
            coords={tracks_dimname: tracks_coord, times_dimname: times_coord},
            dims=(tracks_dimname, times_dimname),
            attrs={
                "long_name": "Flag indicating the status of robust cold pool based on precipitation. 1 = Yes, 0 = No",
                "units": "unitless",
                "_FillValue": fillval,
            }
        )

        # Add new variables to dataset
        dsout["pf_lifetime"] = pf_lifetime
        dsout["cp_status"] = cp_status

    else:
        # Create empty dataset with proper structure
        dsout = ds_cp.isel(tracks=slice(0, 0))  # Empty selection
        tracks_coord = np.array([])
        
        # Create empty DataArrays
        pf_lifetime = xr.DataArray(
            np.array([]),
            coords={tracks_dimname: tracks_coord},
            dims=(tracks_dimname),
            attrs={
                "long_name": "Cold pool lifetime when precipitation is present",
                "units": "hour",
            }
        )
        cp_status = xr.DataArray(
            np.empty((0, ntimes), dtype=int),
            coords={tracks_dimname: tracks_coord, times_dimname: ds_cp[times_dimname]},
            dims=(tracks_dimname, times_dimname),
            attrs={
                "long_name": "Flag indicating the status of robust cold pool based on precipitation. 1 = Yes, 0 = No",
                "units": "unitless",
                "_FillValue": fillval,
            }
        )

        # Add new variables to dataset
        dsout["pf_lifetime"] = pf_lifetime
        dsout["cp_status"] = cp_status

    # Update global attributes
    dsout.attrs["CP_PF_min_area_thresh"] = cp_pf_min_area_thresh
    dsout.attrs["CP_PF_duration_thresh"] = cp_pf_duration_thresh
    dsout.attrs["Created_on"] = time.ctime(time.time())

    #########################################################################################
    # Save output to netCDF file
    logger.debug("Saving data")
    logger.debug(time.ctime())

    # Delete file if it already exists
    if os.path.isfile(statistics_outfile):
        os.remove(statistics_outfile)

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}

    # Write to netcdf file
    dsout.to_netcdf(path=statistics_outfile, mode="w",
                    format="NETCDF4", unlimited_dims=tracks_dimname, encoding=encoding)
    logger.info(f"{statistics_outfile}")

    return statistics_outfile