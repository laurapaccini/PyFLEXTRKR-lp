import numpy as np
import os
import sys
import xarray as xr
import time
import logging
import dask
from dask.distributed import wait
from pyflextrkr.ft_utilities import subset_files_timerange
from pyflextrkr.matchpf_cp_func import matchpf_cp_singlefile

def match_pf_cp_tracks(config):
    """
    Match cold pool tracks with precipitation to filter and calculate statistics.

    Args:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        statistics_outfile: string
            Cold pool track statistics file with precipitation data.
    """

    feature_type = config["feature_type"]
    trackstats_filebase = config["finalstats_filebase"] #old trackstats_filebase
    pfstats_filebase = config.get("pfstats_filebase", "cpwpf_stats_")
    stats_path = config["stats_outpath"]
    tracking_outpath = config["tracking_outpath"]
    cloudid_filebase = config["cloudid_filebase"]  # Keeping variable name for compatibility
    startdate = config["startdate"]
    enddate = config["enddate"]
    tracks_dimname = config["tracks_dimname"]
    times_dimname = config["times_dimname"]
    run_parallel = config["run_parallel"]
    fillval = config["fillval"]
    # Minimum time difference threshold [second] to match track stats and cloudid pixel files
    match_pixel_dt_thresh = config["match_pixel_dt_thresh"]

    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)
    logger.info("Matching cold pool tracks with precipitation for filtering and statistics")

    # Output stats file name
    statistics_outfile = f"{stats_path}{pfstats_filebase}{startdate}_{enddate}.nc"

    #########################################################################################
    # Load track stats
    logger.debug("Loading track statistics data")

    trackstats_file = f"{stats_path}{trackstats_filebase}{startdate}_{enddate}.nc"
    ds = xr.open_dataset(trackstats_file,
                         mask_and_scale=False,
                         decode_times=False)
    ntracks = ds.sizes[tracks_dimname]
    nmaxlength = ds.sizes[times_dimname]
    basetime = ds["base_time"].values
    feature_number = ds["cloudnumber"].values  # Keep variable name for compatibility
    merge_feature_number = ds["merge_cloudnumber"].values
    split_feature_number = ds["split_cloudnumber"].values

    #########################################################################################
    # Find feature files and get their basetime
    infiles_info = subset_files_timerange(
        tracking_outpath,
        cloudid_filebase,
        config["start_basetime"],
        config["end_basetime"],
    )
    feature_file_list = infiles_info[0]
    feature_file_basetime = infiles_info[1]
    nfiles = len(feature_file_list)

    #########################################################################################
    # Find precipitation within each feature
    logger.debug(("Total Number of Tracks:" + str(ntracks)))
    logger.debug("Looping over each pixel file")
    logger.debug((time.ctime()))

    # Create a list to store matchindices for each pixel file
    trackindices_all = []
    timeindices_all = []
    results = []

    # Loop over each pixel file to calculate PF statistics
    for ifile in range(nfiles):
        filename = feature_file_list[ifile]

        # Find all matching time indices from track stats file to the current feature file
        matchindices = np.array(np.where(np.abs(basetime - feature_file_basetime[ifile]) < match_pixel_dt_thresh))
        # The returned match indices are for [tracks, times] dimensions respectively
        idx_track = matchindices[0]
        idx_time = matchindices[1]

        # Get feature numbers for this time (file)
        file_feature_number = feature_number[idx_track, idx_time]
        file_merge_feature_number = merge_feature_number[idx_track, idx_time, :]
        file_split_feature_number = split_feature_number[idx_track, idx_time, :]

        # Save matchindices for the current pixel file to the overall list
        trackindices_all.append(idx_track)
        timeindices_all.append(idx_time)

        # Call function to calculate precipitation stats
        # Serial
        if run_parallel == 0:
            result = matchpf_cp_singlefile(
                filename,
                file_feature_number,
                file_merge_feature_number,
                file_split_feature_number,
                config,
            )
            results.append(result)
        # Parallel
        elif run_parallel >= 1:
            result = dask.delayed(matchpf_cp_singlefile)(
                filename,
                file_feature_number,
                file_merge_feature_number,
                file_split_feature_number,
                config,
            )
            results.append(result)
        else:
            sys.exit('Valid parallelization flag not provided.')

    if run_parallel == 0:
        final_result = results
    elif run_parallel >= 1:
        # Trigger dask computation
        final_result = dask.compute(*results)
        wait(final_result)
    else:
        sys.exit('Valid parallelization flag not provided.')

    #########################################################################################
    # Create arrays to store output
    logger.debug("Collecting track precipitation statistics.")

    maxtracklength = nmaxlength
    numtracks = ntracks

    # Make a variable list and get attributes from one of the returned dictionaries
    # Loop over each return results till one that is not None
    counter = 0
    while counter < nfiles:
        if final_result[counter] is not None:
            var_names = list(final_result[counter][0].keys())
            # Get variable attributes
            var_attrs = final_result[counter][1]
            var_names_2d = final_result[counter][2]
            break
        counter += 1

    # Loop over variable list to create the dictionary entry
    pf_dict = {}
    pf_dict_attrs = {}
    for ivar in var_names:
        if ivar in var_names_2d:
            pf_dict[ivar] = np.full((numtracks, maxtracklength), np.nan, dtype=np.float32)
        else:
            # For any 3D variables, though we don't expect any here
            pf_dict[ivar] = np.full((numtracks, maxtracklength, 1), np.nan, dtype=np.float32)
        pf_dict_attrs[ivar] = var_attrs[ivar]

    # Collect results
    for ifile in range(nfiles):
        if final_result[ifile] is not None:
            # Get the return results for this pixel file
            iResult = final_result[ifile][0]

            # Get trackindices and timeindices for this file
            trackindices = trackindices_all[ifile]
            timeindices = timeindices_all[ifile]

            # Loop over each variable and assign values to output dictionary
            for ivar in var_names:
                if ivar in var_names_2d:
                    pf_dict[ivar][trackindices, timeindices] = iResult[ivar]
                else:
                    # For any 3D variables
                    pf_dict[ivar][trackindices, timeindices, 0] = iResult[ivar]

    # Define a dataset containing all PF variables
    varlist = {}
    # Define output variable dictionary
    for key, value in pf_dict.items():
        if key in var_names_2d:
            varlist[key] = ([tracks_dimname, times_dimname], value, pf_dict_attrs[key])
        else:
            # For any 3D variables
            varlist[key] = ([tracks_dimname, times_dimname, "pf_index"], value, pf_dict_attrs[key])

    # Define coordinate list
    coordlist = {
        tracks_dimname: ([tracks_dimname], np.arange(0, numtracks)),
        times_dimname: ([times_dimname], np.arange(0, maxtracklength)),
    }
    if any(key not in var_names_2d for key in var_names):
        coordlist["pf_index"] = (["pf_index"], np.arange(0, 1))

    # Define global attributes
    gattrlist = {
        "PF_rainrate_thresh": config["pf_rr_thres"],
        "pf_coverage_thresh": config.get("pf_coverage_thresh", 0.2),
        # "heavy_rainrate_thresh": config["heavy_rainrate_thresh"],
    }

    # Define output Xarray dataset
    ds_pf = xr.Dataset(varlist, coords=coordlist, attrs=gattrlist)

    # Merge track stats and precipitation datasets
    dsout = xr.merge([ds, ds_pf], compat="override", combine_attrs="no_conflicts")
    # Update time stamp
    dsout.attrs["Created_on"] = time.ctime(time.time())

    #########################################################################################
    # Save output to netCDF file
    logger.debug("Saving data")
    logger.debug((time.ctime()))

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