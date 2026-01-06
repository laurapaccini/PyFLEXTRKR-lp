"""
Plot cold pool tracks on buoyancy field snapshots.

>python plot_subset_coldpool_tracks.py -s STARTDATE -e ENDDATE -c CONFIG.yml
Optional arguments:
-p 0 (serial), 1 (parallel)
--extent lonmin lonmax latmin latmax (subset domain boundary)
--subset 0 (no), 1 (yes) (subset data before plotting)
--figsize width height (figure size in inches)
--output output_directory (output figure directory)
--figbasename figure base name (output figure base name)
"""
__author__ = "zhe.feng@pnnl.gov, laura.paccini@pnnl.gov (modified for cold pools)"
__created_date__ = "23-Aug-2025"

import argparse
import numpy as np
import os, sys
import xarray as xr
import pandas as pd
from scipy.ndimage import binary_erosion, generate_binary_structure, gaussian_filter
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# For non-gui matplotlib back end
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
mpl.use('agg')
import dask
from dask.distributed import Client, LocalCluster
import warnings
warnings.filterwarnings("ignore")
from pyflextrkr.ft_utilities import load_config, subset_files_timerange

#-----------------------------------------------------------------------
def four_floats(value):
    # Split string by ' '
    values = value.split(' ')
    if len(values) != 4:
        raise argparse.ArgumentError
    # Convert list to array and type to float
    values = np.array(values).astype(float)
    return values

def parse_cmd_args():
    # Define and retrieve the command-line arguments...
    parser = argparse.ArgumentParser(
        description="Plot cold pool tracks on buoyancy field snapshots."
    )
    parser.add_argument("-s", "--start", help="first time in time series to plot, format=YYYY-mm-ddTHH:MM:SS", required=True)
    parser.add_argument("-e", "--end", help="last time in time series to plot, format=YYYY-mm-ddTHH:MM:SS", required=True)
    parser.add_argument("-c", "--config", help="yaml config file for tracking", required=True)
    parser.add_argument("-p", "--parallel", help="flag to run in parallel (0:serial, 1:parallel)", type=int, default=0)
    parser.add_argument("--extent", help="map extent (lonmin lonmax latmin latmax)", type=four_floats, action='store', default=None)
    parser.add_argument("--subset", help="flag to subset data (0:no, 1:yes)", type=int, default=0)
    parser.add_argument("--figsize", nargs='+', help="figure size (width, height) in inches", type=float, default=None)
    parser.add_argument("--output", help="ouput directory", default=None)
    parser.add_argument("--figbasename", help="output figure base name", default="")
    parser.add_argument("--trackstats_file", help="Robust cold pool track stats file name", default=None)
    parser.add_argument("--pixel_path", help="Pixel-level tracknumer mask files directory", default=None)
    parser.add_argument("--time_format", help="Pixel-level file datetime format", default=None)
    args = parser.parse_args()

    # Put arguments in a dictionary
    args_dict = {
        'start_datetime': args.start,
        'end_datetime': args.end,
        'run_parallel': args.parallel,
        'config_file': args.config,
        'extent': args.extent,
        'subset': args.subset,
        'figsize': args.figsize,
        'out_dir': args.output,
        'figbasename': args.figbasename,
        'trackstats_file': args.trackstats_file,
        'pixeltracking_path': args.pixel_path,
        'time_format': args.time_format,
    }

    return args_dict

#--------------------------------------------------------------------------
def make_dilation_structure(dilate_radius, dx, dy):
    """
    Make a circular dilation structure

    Args:
        dilate_radius: float
            Dilation radius [kilometer].
        dx: float
            Grid spacing in x-direction [kilometer].
        dy: float
            Grid spacing in y-direction [kilometer].

    Returns:
        struc: np.array
            Dilation structure array.
    """
    # Convert radius to number grids
    rad_gridx = int(dilate_radius / dx)
    rad_gridy = int(dilate_radius / dy)
    xgrd, ygrd = np.ogrid[-rad_gridx:rad_gridx+1, -rad_gridy:rad_gridy+1]
    # Make dilation structure
    strc = xgrd*xgrd + ygrd*ygrd <= (dilate_radius / dx) * (dilate_radius / dy)
    return strc

#-----------------------------------------------------------------------
def label_perimeter(tracknumber, dilationstructure):
    """
    Labels the perimeter on a 2D map from object tracknumber masks.
    """
    # Get unique tracknumbers that is no nan and > 0 (exclude background)
    tracknumber_unique = np.unique(tracknumber[~np.isnan(tracknumber) & (tracknumber > 0)]).astype(np.int32)

    # Make an array to store the perimeter
    tracknumber_perim = np.zeros(tracknumber.shape, dtype=np.int32)

    # Loop over each tracknumbers (excluding 0/background)
    for ii in tracknumber_unique:
        # Isolate the track mask
        itn = tracknumber == ii
        # Erode the mask by 1 pixel
        itn_erode = binary_erosion(itn, structure=dilationstructure).astype(itn.dtype)
        # Subtract the eroded area to get the perimeter
        iperim = np.logical_xor(itn, itn_erode)
        # Label the perimeter pixels with the track number
        tracknumber_perim[iperim == 1] = ii

    return tracknumber_perim

#-----------------------------------------------------------------------
def calc_track_center(tracknumber, longitude, latitude):
    """
    Calculates the center location from labeled tracks.
    """
    
    # Find unique tracknumbers
    tracknumber_uniqe = np.unique(tracknumber[~np.isnan(tracknumber)])
    num_tracknumber = len(tracknumber_uniqe)
    # Make arrays for track center locations
    lon_c = np.full(num_tracknumber, np.nan, dtype=float)
    lat_c = np.full(num_tracknumber, np.nan, dtype=float)

    # Loop over each tracknumbers to calculate the mean lat/lon & x/y for their center locations
    for ii, itn in enumerate(tracknumber_uniqe):
        iyy, ixx = np.where(tracknumber == itn)
        lon_c[ii] = np.mean(longitude[tracknumber == itn])
        lat_c[ii] = np.mean(latitude[tracknumber == itn])
        
    return lon_c, lat_c, tracknumber_uniqe

#-----------------------------------------------------------------------
def get_track_stats(trackstats_file, start_datetime, end_datetime, dt_thres):
    """
    Subset robust cold pool tracks statistics data within start/end datetime

    Args:
        trackstats_file: string
            Robust cold pool track statistics file name.
        start_datetime: string
            Start datetime to subset tracks.
        end_datetime: string
            End datetime to subset tracks.
        dt_thres: timedelta
            A timedelta threshold to retain tracks.
            
    Returns:
        track_dict: dictionary
            Dictionary containing track stats data.
    """
    # Read track stats file
    dss = xr.open_dataset(trackstats_file)
    stats_starttime = dss.base_time.isel(times=0)
    # Convert input datetime to np.datetime64
    stime = np.datetime64(start_datetime)
    etime = np.datetime64(end_datetime)
    time_res = dss.attrs['time_resolution_hour']

    # Find track initiated within the time window
    idx = np.where((stats_starttime >= stime) & (stats_starttime <= etime))[0]
    ntracks = len(idx)
    print(f'Number of robust cold pool tracks within input period: {ntracks}')

    # Calculate track lifetime
    lifetime = dss.track_duration.isel(tracks=idx) * time_res

    # Subset these tracks and put in a dictionary
    track_dict = {
        'ntracks': ntracks,
        'lifetime': lifetime,
        'track_bt': dss['base_time'].isel(tracks=idx),
        'track_meanlon': dss['meanlon'].isel(tracks=idx) * xscale,
        'track_meanlat': dss['meanlat'].isel(tracks=idx) * yscale,
        'dt_thres': dt_thres,
        'time_res': time_res,
    }
    
    return track_dict

#-----------------------------------------------------------------------
def plot_map(pixel_dict, plot_info, map_info, track_dict):
    """
    Plotting function for cold pool tracks on buoyancy field.

    Args:
        pixel_dict: dictionary
            Dictionary containing pixel-level variables
        plot_info: dictionary
            Dictionary containing plotting variables
        map_info: dictionary
            Dictionary containing mapping variables
        track_dict: dictionary
            Dictionary containing tracking variables

    Returns:
        fig: object
            Figure object.
    """
    
    # Get pixel data from dictionary
    pixel_bt = pixel_dict['pixel_bt']
    xx = pixel_dict['longitude']
    yy = pixel_dict['latitude']
    cp_intensity = pixel_dict['cp_intensity']
    cp_depth = pixel_dict['cp_depth']
    int_buoyancy = pixel_dict['int_buoyancy']
    wind_speed = pixel_dict['wind_speed']
    u10 = pixel_dict['u10']
    v10 = pixel_dict['v10']
    pcp = pixel_dict['pcp']
    divergence = pixel_dict['divergence']
    tn_perim = pixel_dict['tn_perim']
    tn = pixel_dict['tn']
    # Get track data from dictionary
    ntracks = track_dict['ntracks']
    lifetime = track_dict['lifetime']
    track_bt = track_dict['track_bt']
    track_meanlon = track_dict['track_meanlon']
    track_meanlat = track_dict['track_meanlat']

    dt_thres = track_dict['dt_thres']
    time_res = track_dict['time_res']
    # Get plot info from dictionary
    fontsize = plot_info['fontsize']
    levels = plot_info['levels']
    cmaps = plot_info['cmap']
    titles = plot_info['titles']
    remove_oob_low = plot_info.get('remove_oob_low', False)
    remove_oob_high = plot_info.get('remove_oob_high', False)
    cblabels = plot_info['cblabels']
    cbticks = plot_info['cbticks']
    marker_size = plot_info['marker_size']
    tracknumber_fontsize = plot_info['tracknumber_fontsize']
    trackpath_linewidth = plot_info['trackpath_linewidth']
    trackpath_color = plot_info['trackpath_color']
    xlabel = plot_info['xlabel']
    ylabel = plot_info['ylabel']
    timestr = plot_info['timestr']
    figname = plot_info['figname']
    figsize = plot_info['figsize']
    mask_alpha = plot_info.get('mask_alpha', 1)
    perim_plot = plot_info.get('perim_plot', 'contour')
    perim_linewidth = plot_info.get('perim_linewidth', None)

    # Map domain, lat/lon ticks, background map features
    map_extent = map_info['map_extent']

    # Time difference matching pixel-time and track time
    dt_match = 1  # [min]

    # Marker style for track center
    marker_style = dict(edgecolor=trackpath_color, facecolor=trackpath_color, linestyle='-', marker='o')

    # Set up figure
    mpl.rcParams['font.size'] = fontsize
    fig = plt.figure(figsize=figsize, dpi=200)

    # Set GridSpec for 2x2 layout with horizontal colorbars below each plot
    # 4 rows (plot1, cbar1, plot2, cbar2) x 2 columns (left, right)
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 0.05, 1, 0.05], width_ratios=[1, 1])
    gs.update(wspace=0.3, hspace=0.3, left=0.08, right=0.95, top=0.90, bottom=0.08)
    
    # Create subplots and colorbars
    ax1 = plt.subplot(gs[0, 0])  # Features + pcp contours + wind vectors (top-left)
    ax2 = plt.subplot(gs[0, 1])  # Integrated buoyancy + wind vectors (top-right)
    ax3 = plt.subplot(gs[2, 0])  # w_500.0 (masked) + wind vectors (bottom-left)
    ax4 = plt.subplot(gs[2, 1])  # Precipitation + wind vectors (bottom-right)
    cax1 = plt.subplot(gs[1, 0])  # Features colorbar (below ax1)
    cax2 = plt.subplot(gs[1, 1])  # Integrated buoyancy colorbar (below ax2)
    cax3 = plt.subplot(gs[3, 0])  # w_500.0 colorbar (below ax3)
    cax4 = plt.subplot(gs[3, 1])  # Precipitation colorbar (below ax4)
    
    # Figure title: time
    fig.suptitle(timestr, fontsize=fontsize*1.2)

    # Define zoom extent for all panels
    zoom_extent = [200, 350, 250, 380]  # [xmin, xmax, ymin, ymax]

    # Prepare wind vectors - subsample for clearer display
    skip = 2  # Plot every 5th vector
    u10_data = u10.data if hasattr(u10, 'data') else u10
    v10_data = v10.data if hasattr(v10, 'data') else v10
    
    # Debug: Check wind data
    # print(f"Wind data shapes: u10={u10_data.shape}, v10={v10_data.shape}")
    # print(f"Wind data ranges: u10=[{np.nanmin(u10_data):.2f}, {np.nanmax(u10_data):.2f}], v10=[{np.nanmin(v10_data):.2f}, {np.nanmax(v10_data):.2f}]")
    # print(f"Wind data non-zero count: u10={np.count_nonzero(~np.isnan(u10_data))}, v10={np.count_nonzero(~np.isnan(v10_data))}")
    # print(f"Coordinate ranges: xx=[{np.nanmin(xx):.1f}, {np.nanmax(xx):.1f}], yy=[{np.nanmin(yy):.1f}, {np.nanmax(yy):.1f}]")
    
    #################################################################
    # PANEL 1: Features mask + smoothed 2 mm/hr precip contours + wind vectors
    #################################################################
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title('Cold Pool Features', loc='left', fontsize=fontsize)
    
    # Plot features mask in colors
    tn_data = tn.data.copy()
    tn_masked = np.ma.masked_where(tn_data <= 0, tn_data)
    
    if not np.ma.is_masked(tn_masked) or not tn_masked.mask.all():
        cmap_features = plt.get_cmap(cmaps['features_cmap'])
        unique_features = np.unique(tn_masked.compressed())
        if len(unique_features) > 0:
            norm_features = mpl.colors.BoundaryNorm(np.arange(0.5, len(unique_features)+1.5, 1), ncolors=cmap_features.N, clip=True)
            cf1 = ax1.pcolormesh(xx, yy, tn_masked, norm=norm_features, cmap=cmap_features, zorder=2)
        else:
            cf1 = None
    else:
        cf1 = None
    
    # Add smoothed 2 mm/hr precipitation contours
    pcp_smoothed = gaussian_filter(pcp, sigma=2)  # Smooth with sigma=2
    pcp_masked_contour = np.ma.masked_where(pcp_smoothed < 2.0, pcp_smoothed)
    if not np.ma.is_masked(pcp_masked_contour) or not pcp_masked_contour.mask.all():
        cs1 = ax1.contour(xx, yy, pcp_smoothed, levels=[2.0], colors='black', 
                         linewidths=2.0, linestyles='solid', zorder=3)
    

    pcp_masked_contour5 = np.ma.masked_where(pcp_smoothed < 5.0, pcp_smoothed)
    if not np.ma.is_masked(pcp_masked_contour5) or not pcp_masked_contour5.mask.all():
        cs1 = ax1.contour(xx, yy, pcp_smoothed, levels=[5.0], colors='red',
                         linewidths=2.0, linestyles='solid', zorder=3)
    
    # Add wind vectors
    q1 = ax1.quiver(xx[::skip, ::skip], yy[::skip, ::skip], 
                    u10_data[::skip, ::skip], v10_data[::skip, ::skip],
                    scale=120, width=0.004, color='black', alpha=0.8, zorder=4)
    ax1.quiverkey(q1, X=0.85, Y=1.05, U=5, label='5 m/s', labelpos='E', coordinates='axes')
    
    ax1.set_xlim(zoom_extent[0], zoom_extent[1])
    ax1.set_ylim(zoom_extent[2], zoom_extent[3])
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    
    # Features colorbar
    if cf1 is not None:
        cb1 = plt.colorbar(cf1, cax=cax1, #label='Feature ID',
                           extend='neither', orientation='horizontal')
    
    #################################################################
    # PANEL 2: Integrated buoyancy + wind vectors
    #################################################################
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_title('Integrated Buoyancy', loc='left', fontsize=fontsize)
    
    # Plot integrated buoyancy
    cmap_buoyancy = plt.get_cmap('Blues_r')
    int_buoyancy_plot = int_buoyancy.copy()
    int_buoyancy_plot = np.ma.masked_where(int_buoyancy_plot >= 0, int_buoyancy_plot)  # Mask non-negative values
    
    cf2 = ax2.pcolormesh(xx, yy, int_buoyancy_plot, cmap=cmap_buoyancy, 
                         vmin=-5, vmax=0, zorder=2)
    
    # Add smoothed 2 mm/hr precipitation contours
    if not np.ma.is_masked(pcp_masked_contour) or not pcp_masked_contour.mask.all():
        cs1 = ax2.contour(xx, yy, pcp_smoothed, levels=[2.0], colors='black', 
                         linewidths=2.0, linestyles='solid', zorder=3)
    if not np.ma.is_masked(pcp_masked_contour5) or not pcp_masked_contour5.mask.all():
        cs1 = ax2.contour(xx, yy, pcp_smoothed, levels=[5.0], colors='red',
                         linewidths=2.0, linestyles='solid', zorder=3)
        
    # Add wind vectors
    q2 = ax2.quiver(xx[::skip, ::skip], yy[::skip, ::skip], 
                    u10_data[::skip, ::skip], v10_data[::skip, ::skip],
                    scale=120, width=0.004, color='black', alpha=0.8, zorder=4)
    ax2.quiverkey(q2, X=0.85, Y=1.05, U=5, label='5 m/s', labelpos='E', coordinates='axes')
    
    ax2.set_xlim(zoom_extent[0], zoom_extent[1])
    ax2.set_ylim(zoom_extent[2], zoom_extent[3])
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    
    # Integrated buoyancy colorbar
    cb2 = plt.colorbar(cf2, cax=cax2, #label='Integrated Buoyancy (m²/s²)', 
                       extend='min', orientation='horizontal')
    
    #################################################################
    # PANEL 3: w_500.0 (masked -0.05 to 0.05) + wind vectors
    #################################################################
    ax3.set_aspect('equal', adjustable='box')
    ax3.set_title('Vertical Velocity at 500m', loc='left', fontsize=fontsize)
    
    # Plot w_500.0 with masking of small values
    wind_plot = wind_speed.data if hasattr(wind_speed, 'data') else wind_speed
    wind_masked = np.ma.masked_where(np.abs(wind_plot) <= 0.05, wind_plot)
    
    cmap_w = plt.get_cmap('RdBu_r')
    cf3 = ax3.pcolormesh(xx, yy, wind_masked, cmap=cmap_w, 
                         vmin=-1, vmax=1, zorder=2)
    
    # Add smoothed 2 mm/hr precipitation contours
    if not np.ma.is_masked(pcp_masked_contour) or not pcp_masked_contour.mask.all():
        cs1 = ax3.contour(xx, yy, pcp_smoothed, levels=[2.0], colors='black', 
                         linewidths=2.0, linestyles='solid', zorder=3)
    if not np.ma.is_masked(pcp_masked_contour5) or not pcp_masked_contour5.mask.all():
        cs1 = ax3.contour(xx, yy, pcp_smoothed, levels=[5.0], colors='gray',
                         linewidths=2.0, linestyles='solid', zorder=3)

    # Add wind vectors
    q3 = ax3.quiver(xx[::skip, ::skip], yy[::skip, ::skip], 
                    u10_data[::skip, ::skip], v10_data[::skip, ::skip],
                    scale=120, width=0.004, color='black', alpha=0.8, zorder=4)
    ax3.quiverkey(q3, X=0.85, Y=1.05, U=5, label='5 m/s', labelpos='E', coordinates='axes')
    
    ax3.set_xlim(zoom_extent[0], zoom_extent[1])
    ax3.set_ylim(zoom_extent[2], zoom_extent[3])
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel(ylabel)
    
    # w_500.0 colorbar
    cb3 = plt.colorbar(cf3, cax=cax3, label='w at 500m (m/s)', 
                       extend='both', orientation='horizontal')

    #################################################################
    # PANEL 4: Surface wind divergence + feature contours
    #################################################################
    ax4.set_aspect('equal', adjustable='box')
    ax4.set_title('Surface Wind Divergence', loc='left', fontsize=fontsize)
    
    # Plot surface wind divergence in colors
    cmap_div = plt.get_cmap('RdBu_r')  # Red for convergence (negative), Blue for divergence (positive)
    div_plot = divergence.copy()
    # Mask very small values for clarity
    div_plot = np.ma.masked_where(np.abs(div_plot) < 1e-5, div_plot)
    
    # Convert from 1/s to 10^-4 /s for better visualization
    div_plot_scaled = div_plot * 1e4
    
    cf4 = ax4.pcolormesh(xx, yy, div_plot_scaled, cmap=cmap_div, 
                         vmin=-5, vmax=5, zorder=2)
    
    # Add features contours in black
    if not np.ma.is_masked(tn_masked) or not tn_masked.mask.all():
        # Create binary mask for all features
        feature_mask = np.zeros_like(tn_data)
        feature_mask[tn_data > 0] = 1
        
        if np.any(feature_mask > 0):
            cs4 = ax4.contour(xx, yy, feature_mask, levels=[0.5], colors='black', 
                             linewidths=2, zorder=3)
    
    ax4.set_xlim(zoom_extent[0], zoom_extent[1])
    ax4.set_ylim(zoom_extent[2], zoom_extent[3])
    ax4.set_xlabel(xlabel)
    ax4.set_ylabel(ylabel)
    
    # Divergence colorbar
    cb4 = plt.colorbar(cf4, cax=cax4, label='Divergence (10⁻⁴ s⁻¹)', 
                       extend='both', orientation='horizontal')

    # Get domain maximum values 
    domain_max_x = map_extent[1] - map_extent[0]
    domain_max_y = map_extent[3] - map_extent[2]

    # Plot track centroids and paths on all panels
    axes_list = [ax1, ax2, ax3, ax4]  # Plot tracks on all four panels
    
    for itrack in range(0, ntracks):
        # Get duration of the track
        ilifetime = lifetime.values[itrack]
        itracknum = lifetime.tracks.data[itrack]+1
        idur = (ilifetime / time_res).astype(int)
        
        # Get basetime of the track and the last time
        ibt = track_bt.values[itrack,:idur]
        ibt_end = np.nanmax(ibt)
        # Compute time difference between current pixel-level data time and the last time of the track
        idt_end = (pixel_bt - ibt_end).astype('timedelta64[m]')
        # Proceed if time difference is <= threshold
        # This means for tracks that end longer than the time threshold are not plotted
        if (idt_end <= dt_thres):
            # Find times in track data <= current pixel-level file time
            idx_cut = np.where(ibt <= pixel_bt)[0]
            idur_cut = len(idx_cut)
            if (idur_cut > 0):
                ### Handle tracks that cross domain for plotting
                # Get adjusted positions
                adjusted_lon = track_meanlon.values[itrack, idx_cut]
                adjusted_lat = track_meanlat.values[itrack, idx_cut]

                # Wrap positions back into domain for plotting
                wrapped_lon = np.mod(adjusted_lon - map_extent[0], domain_max_x) + map_extent[0]
                wrapped_lat = np.mod(adjusted_lat - map_extent[2], domain_max_y) + map_extent[2]

                # Identify where wrap-around occurs to split the trajectory
                lon_diff = np.abs(np.diff(wrapped_lon))
                lat_diff = np.abs(np.diff(wrapped_lat))
                wrap_indices = np.where((lon_diff > (domain_max_x / 2)) | (lat_diff > (domain_max_y / 2)))[0] + 1

                # Split the trajectory at wrap-around points
                split_lon = np.split(wrapped_lon, wrap_indices)
                split_lat = np.split(wrapped_lat, wrap_indices)

                # Plot each segment separately on all panels
                for ax in axes_list:
                    for lon_seg, lat_seg in zip(split_lon, split_lat):
                        # Ensure there are at least two points to plot a line
                        if len(lon_seg) >= 2:
                            ax.plot(lon_seg, lat_seg, lw=trackpath_linewidth, ls='-', color=trackpath_color, zorder=4)
                        else:
                            # For single points, plot as markers
                            ax.scatter(lon_seg, lat_seg, s=marker_size, color=trackpath_color, zorder=4)

                    # Initiation location (adjusted)
                    init_lon = track_meanlon.values[itrack, 0]
                    init_lat = track_meanlat.values[itrack, 0]
                    wrapped_init_lon = np.mod(init_lon - map_extent[0], domain_max_x) + map_extent[0]
                    wrapped_init_lat = np.mod(init_lat - map_extent[2], domain_max_y) + map_extent[2]
                    ax.scatter(wrapped_init_lon, wrapped_init_lat, s=marker_size*2, zorder=5, **marker_style)
                
        # Find the closest time from track times
        idt = np.abs((ibt - pixel_bt).astype('timedelta64[m]'))
        idx_match = np.argmin(idt)
        idt_match = idt[idx_match]
        # Get track data
        _imeanlon = track_meanlon.data[itrack,idx_match]
        _imeanlat = track_meanlat.data[itrack,idx_match]

        # Adjust positions for plotting
        _iwrapped_meanlon = np.mod(_imeanlon - map_extent[0], domain_max_x) + map_extent[0]
        _iwrapped_meanlat = np.mod(_imeanlat - map_extent[2], domain_max_y) + map_extent[2]
  
        # Proceed if time difference is < dt_match
        if (idt_match < dt_match):
            # Overplot tracknumbers at current frame on all panels (only if within zoom extent)
            if (_iwrapped_meanlon > zoom_extent[0]) & (_iwrapped_meanlon < zoom_extent[1]) & \
                (_iwrapped_meanlat > zoom_extent[2]) & (_iwrapped_meanlat < zoom_extent[3]):
                for ax in axes_list:
                    ax.text(_iwrapped_meanlon+0.02, _iwrapped_meanlat+0.02, f'{itracknum:.0f}',
                            color='r', size=tracknumber_fontsize, weight='bold', ha='left', va='center', zorder=5)
  
    # Thread-safe figure output
    canvas = FigureCanvas(fig)
    canvas.print_png(figname)
    fig.savefig(figname)
    
    return fig

#-----------------------------------------------------------------------
def work_for_time_loop(datafile, track_dict, map_info, plot_info, config):
    """
    Process data for a single frame and make the plot.

    Args:
        datafile: string
            Pixel-level data filename
        track_dict: dictionary
            Dictionary containing tracking variables
        map_info: dictionary
            Dictionary containing mapping variables
        plot_info: dictionary
            Dictionary containing plotting variables
        config: dictionary
            Dictionary containing config parameters

    Returns:
        1.
    """
    
    map_extent = map_info.get('map_extent', None)
    perim_thick = plot_info.get('perim_thick')
    figdir = plot_info.get('figdir')
    figbasename = plot_info.get('figbasename')

    # Read pixel-level data
    ds = xr.open_dataset(datafile, mask_and_scale=False)
    pixel_bt = ds.time.data

    # Get map extent from data
    if map_extent is None:
        lonmin = ds['longitude'].min().item()
        lonmax = ds['longitude'].max().item()
        latmin = ds['latitude'].min().item()
        latmax = ds['latitude'].max().item()
        map_extent = [lonmin, lonmax, latmin, latmax]
        map_info['map_extent'] = map_extent
        map_info['subset'] = subset
    
    dilationstructure = make_dilation_structure(perim_thick, pixel_radius, pixel_radius)

    # Use robust cold pool mask from 'tracknumber' in cptracks_*.nc
    if 'tracknumber' in ds:
        robust_mask = ds['tracknumber']
    else:
        raise ValueError("tracknumber variable not found in file. Are you using cptracks_*.nc files?")

    # Subset pixel data within the map domain
    if subset == 1:
        map_extent = map_info['map_extent']
        buffer = 0.0  # buffer area for subset
        lonmin, lonmax = map_extent[0]-buffer, map_extent[1]+buffer
        latmin, latmax = map_extent[2]-buffer, map_extent[3]+buffer
        mask = (ds['longitude'] >= lonmin) & (ds['longitude'] <= lonmax) & \
               (ds['latitude'] >= latmin) & (ds['latitude'] <= latmax)
        
        # Apply mask to all variables consistently
        cp_intensity_sub = ds['cp_intensity'].where(mask == True, drop=True).squeeze()
        cp_depth_sub = ds['cp_depth'].where(mask == True, drop=True).squeeze()
        pcp_sub = ds['precipitation'].where(mask == True, drop=True).squeeze()
        wind_sub = ds['w_500.0'].where(mask == True, drop=True).squeeze()
        qv2_sub = ds['qv2'].where(mask == True, drop=True).squeeze()
        u10_sub = ds['u10'].where(mask == True, drop=True).squeeze()
        v10_sub = ds['v10'].where(mask == True, drop=True).squeeze()
        tracknumber_sub = robust_mask.where(mask == True, drop=True).squeeze()
        
        # Get coordinates from the original dataset after applying the same mask
        lon_sub = ds['longitude'].where(mask == True, drop=True).data
        lat_sub = ds['latitude'].where(mask == True, drop=True).data
    else:
        cp_intensity_sub = ds['cp_intensity'].squeeze()
        cp_depth_sub = ds['cp_depth'].squeeze()
        pcp_sub = ds['precipitation'].squeeze()
        wind_sub = ds['w_500.0'].squeeze()
        qv2_sub = ds['qv2'].squeeze()
        u10_sub = ds['u10'].squeeze()
        v10_sub = ds['v10'].squeeze()
        tracknumber_sub = robust_mask.squeeze()
        
        # Get coordinates from the original dataset
        lon_sub = ds['longitude'].data
        lat_sub = ds['latitude'].data

    # Calculate integrated buoyancy: int_buoyancy = (cp_intensity**2)/(-2)
    int_buoyancy_sub = (cp_intensity_sub.data**2) / (-2.0)

    # Calculate surface wind divergence: div = du/dx + dv/dy
    # Extract u10 and v10 data
    u10_data = u10_sub.data if hasattr(u10_sub, 'data') else u10_sub
    v10_data = v10_sub.data if hasattr(v10_sub, 'data') else v10_sub
    
    # Calculate gradients (assuming lon_sub and lat_sub are in meters before scaling)
    # Grid spacing in meters
    dx = np.diff(lon_sub).mean() if lon_sub.ndim == 1 else np.diff(lon_sub[0, :]).mean()
    dy = np.diff(lat_sub).mean() if lat_sub.ndim == 1 else np.diff(lat_sub[:, 0]).mean()
    
    # Calculate derivatives using central differences
    du_dx = np.gradient(u10_data, dx, axis=1)  # Derivative along x (columns)
    dv_dy = np.gradient(v10_data, dy, axis=0)  # Derivative along y (rows)
    
    # Calculate divergence
    divergence_sub = du_dx + dv_dy

    # Scale x, y (change units from [m] to [km])
    lon_sub = lon_sub * xscale
    lat_sub = lat_sub * yscale
    
    # Check if coordinates are already 2D grids or 1D arrays
    if lon_sub.ndim == 1 and lat_sub.ndim == 1:
        # Create 2D coordinate grids for plotting if they are 1D
        # Use indexing='xy' to ensure proper orientation
        xx, yy = np.meshgrid(lon_sub, lat_sub, indexing='xy')
    else:
        # Use coordinates as they are if already 2D
        xx, yy = lon_sub, lat_sub
    
    # Ensure all data arrays have the same shape as coordinate grids
    # This helps prevent flipping issues
    if hasattr(pcp_sub, 'data'):
        pcp_data = pcp_sub.data
    else:
        pcp_data = pcp_sub
        
    # # Intentionally flip only the precipitation data
    # if pcp_data.shape == xx.shape:
    #     # If shapes match, flip the precipitation data
    #     pcp_data = np.flipud(pcp_data)
    #     print(f"Intentionally flipped precipitation data with shape {pcp_data.shape}")
    # elif pcp_data.shape == (xx.shape[1], xx.shape[0]):
    #     # If transposition is needed, transpose first then flip
    #     pcp_data = pcp_data.T
    #     pcp_data = np.flipud(pcp_data)
    #     print(f"Transposed and flipped precipitation data from {pcp_sub.shape} to {pcp_data.shape}")
    # else:
    #     print(f"Warning: precipitation shape {pcp_data.shape} doesn't match coordinates {xx.shape}")
    
    # Update the precipitation data
    pcp = pcp_data

    # Get object perimeters
    tn_perim = label_perimeter(tracknumber_sub.data, dilationstructure)

    # Calculates track center locations
    # Use 1D coordinate arrays for track center calculation if coordinates are 2D
    if lon_sub.ndim == 2:
        lon_1d = lon_sub[0, :]  # First row for longitude
        lat_1d = lat_sub[:, 0]  # First column for latitude
    else:
        lon_1d = lon_sub
        lat_1d = lat_sub
    lon_tn, lat_tn, tn_unique = calc_track_center(tracknumber_sub.data, xx, yy)

    # Plotting variables
    timestr = ds['time'].squeeze().dt.strftime("%Y-%m-%d %H:%M:%S UTC").data
    fignametimestr = ds['time'].squeeze().dt.strftime("%Y%m%d_%H%M%S").data.item()
    figname = f'{figdir}{figbasename}{fignametimestr}.png'

    # Put variables in dictionaries
    pixel_dict = {
        'pixel_bt': pixel_bt,
        'longitude': xx,  # 2D coordinate grid
        'latitude': yy,   # 2D coordinate grid
        'cp_intensity': cp_intensity_sub,
        'cp_depth': cp_depth_sub,
        'int_buoyancy': int_buoyancy_sub,
        'wind_speed': wind_sub,
        'u10': u10_sub,
        'v10': v10_sub,
        'qv2': qv2_sub,
        'pcp': pcp,  # Use corrected precipitation data
        'divergence': divergence_sub,  # Surface wind divergence
        'tn': tracknumber_sub,
        'tn_perim': tn_perim,
        'lon_tn': lon_tn,
        'lat_tn': lat_tn,
        'tracknumber_unique': tn_unique,
    }
    plot_info['timestr'] = timestr
    plot_info['figname'] = figname
    # Call plotting function
    fig = plot_map(pixel_dict, plot_info, map_info, track_dict)
    plt.close(fig)
    print(figname)

    ds.close()
    return 1


if __name__ == "__main__":

    # Get the command-line arguments...
    args_dict = parse_cmd_args()
    start_datetime = args_dict.get('start_datetime')
    end_datetime = args_dict.get('end_datetime')
    run_parallel = args_dict.get('run_parallel')
    config_file = args_dict.get('config_file')
    map_extent = args_dict.get('extent')
    subset = args_dict.get('subset')
    figsize = args_dict.get('figsize')
    out_dir = args_dict.get('out_dir')
    figbasename = args_dict.get('figbasename')
    trackstats_file = args_dict.get('trackstats_file')
    pixeltracking_path = args_dict.get('pixeltracking_path')
    time_format = args_dict.get('time_format')

    if time_format is None: time_format = "yyyymodd_hhmmss"

    # Determine the figsize based on lat/lon ratio
    if (figsize is None):
        # If map_extent is specified, calculate aspect ratio
        if (map_extent is not None):
            # Get map aspect ratio from map_extent (minlon, maxlon, minlat, maxlat)
            lon_span = map_extent[1] - map_extent[0]
            lat_span = map_extent[3] - map_extent[2]
            fig_ratio_yx = lat_span / lon_span

            figsize_x = 12
            figsize_y = figsize_x * fig_ratio_yx
            figsize_y = float("{:.2f}".format(figsize_y))  # round to 2 decimal digits
            figsize = [figsize_x, figsize_y]
        else:
            figsize = [10, 10]

    # Specify plotting info for cold pools
    # Precipitation contour levels (only 2 and 5 mm/hr)
    pcp_contours = [2, 5]
    # Features mask color levels (we'll use discrete colors for each feature)
    features_levels = np.arange(0, 101, 1)  # Will be adjusted based on actual features
    features_ticks = np.arange(0, 101, 10)
    # Integrated buoyancy color levels (-5 to 0)
    int_buoyancy_levels = np.linspace(-5, 0, 21)
    int_buoyancy_ticks = np.arange(-5, 1, 1)
    # Wind speed color levels (-0.2 to 0.2 m/s)
    wind_levels = np.linspace(-0.2, 0.2, 21)
    wind_ticks = np.arange(-0.2, 0.3, 0.1)
    # QV2 color levels ( g/g)
    qv2_levels = np.linspace(-7.5/1e3, 7.5/1e3, 21)
    qv2_ticks = np.arange(-7.5/1e3, 7.6/1e3, 5/1e3)

    levels = {
        'pcp_contours': pcp_contours,
        'features_levels': features_levels,
        'int_buoyancy_levels': int_buoyancy_levels,
        'wind_levels': wind_levels,
        'qv2_levels': qv2_levels,
    }
    # Colorbar ticks & labels
    cbticks = {
        'features_ticks': features_ticks,
        'int_buoyancy_ticks': int_buoyancy_ticks,
        'wind_ticks': wind_ticks,
        'qv2_ticks': qv2_ticks,
    }
    cblabels = {
        'features_label': 'Feature ID',
        'int_buoyancy_label': 'Integrated Buoyancy (m²/s²)',
        'wind_label': 'Vertical Wind Speed at 500m (m/s)',
        'qv2_label': 'Specific humidity at 2m (g/g)'
    }
    # Colormaps
    features_cmap = 'tab20'  # Good for discrete features
    int_buoyancy_cmap = 'RdBu_r'
    wind_cmap = 'RdBu_r'
    qv2_cmap = 'BrBG'
    cmaps = {
        'features_cmap': features_cmap,
        'int_buoyancy_cmap': int_buoyancy_cmap,
        'wind_cmap': wind_cmap,
        'qv2_cmap': qv2_cmap,
    }
    titles = {
        'features_title': 'Cold Pool Features with Precipitation Contours',
        'int_buoyancy_title': 'Integrated Buoyancy with Precipitation Contours',
        'wind_title': 'Vertical Wind Speed at 500m with Feature Contours',
        'qv2_title': 'Specific Humidity at 2m with Feature Contours'
    }
    
    # Scaling factor for x, y coordinates
    xscale = 1e-3
    yscale = 1e-3

    plot_info = {
        'fontsize': 14,     # plot font size
        'cmap': cmaps,
        'levels': levels,
        'cbticks': cbticks, 
        'cblabels': cblabels,
        'titles': titles,
        'cp_alpha': 0.7,
        'pcp_alpha': 0.9,
        'remove_oob_low': True,   # mask out-of-bounds low values (< min(levels))
        'remove_oob_high': False,  # mask out-of-bounds high values (> max(levels))
        'mask_alpha': 0.6,   # transparancy alpha for perimeter mask
        'marker_size': 10,   # track symbol marker size
        'tracknumber_fontsize': 14,
        'perim_plot': 'contour',  # method to plot tracked feature perimeter ('contour', 'pcolormesh')
        'perim_linewidth': 3.0,  # perimeter line width for 'contour' method
        'perim_thick': 2,  # width of the tracked feature perimeter [km]
        'trackpath_linewidth': 1.5, # track path line width
        'trackpath_color': 'blueviolet',    # track path color
        'xlabel': 'X (km)',
        'ylabel': 'Y (km)',
        'figsize': figsize,
        'figbasename': figbasename,
    }

    # Customize lat/lon labels
    lonv = None
    latv = None
    # Put map info in a dictionary
    map_info = {
        'map_extent': map_extent,
        'subset': subset,
        'lonv': lonv,
        'latv': latv,
        'draw_land': False,
        'draw_border': False,
        'draw_state': False,
    }

    # Load config and get paths
    config = load_config(config_file)
    stats_path = config["stats_outpath"]
    pixeltracking_path = config["pixeltracking_outpath"]
    pixeltracking_filebase = config["pixeltracking_filebase"]
    cprobust_filebase = config.get("cprobust_filebase", "cp_tracks_robust_")
    startdate = config["startdate"]
    enddate = config["enddate"]
    n_workers = config["nprocesses"]
    datatimeresolution = config["datatimeresolution"]  # hour
    pixel_radius = config["pixel_radius"]

    # Tracks that end longer than this threshold from the current pixel-level frame are not plotted
    # This treshold controls the time window to retain previous tracks
    track_retain_time_min = (datatimeresolution * 60)

    # Create a timedelta threshold in minutes
    dt_thres = datetime.timedelta(minutes=track_retain_time_min)

    # If trackstats_file is not specified, use robust cold pool tracks
    if trackstats_file is None:
        trackstats_file = f"{stats_path}{cprobust_filebase}{startdate}_{enddate}.nc"
    
    if pixeltracking_path is None:
        pixeltracking_path = f"{config['root_path']}{config['pixel_path_name']}/{startdate}_{enddate}/"

    # Output figure directory
    if out_dir is None:
        figdir = f'{pixeltracking_path}quicklooks_coldpool_tracks/'
    else:
        figdir = out_dir
    os.makedirs(figdir, exist_ok=True)
    # Add to plot_info dictionary
    plot_info['figdir'] = figdir

    # Convert datetime string to Epoch time (base time)
    start_basetime = pd.to_datetime(start_datetime).timestamp()
    end_basetime = pd.to_datetime(end_datetime).timestamp()
    # Subtract start_datetime by TimeDelta to include tracks
    # that start before the start_datetime but may not have ended yet
    TimeDelta = pd.Timedelta(days=30)
    start_datetime_4stats = (pd.to_datetime(start_datetime) - TimeDelta).strftime('%Y-%m-%dT%H:%M:%S')

    # Find all pixel-level files that match the input datetime
    datafiles, \
    datafiles_basetime, \
    datafiles_datestring, \
    datafiles_timestring = subset_files_timerange(
        pixeltracking_path,
        pixeltracking_filebase,
        start_basetime,
        end_basetime,
        time_format="yyyymodd_hhmmss",
    )
    print(f'Number of pixel files: {len(datafiles)}')

    # Get robust cold pool track stats data
    track_dict = get_track_stats(trackstats_file, start_datetime_4stats, end_datetime, dt_thres)

    # Serial option
    if run_parallel == 0:
        for ifile in range(len(datafiles)):
            print(datafiles[ifile])
            result = work_for_time_loop(datafiles[ifile], track_dict, map_info, plot_info, config)

    # Parallel option
    elif run_parallel == 1:
        # Set Dask temporary directory for workers
        dask_tmp_dir = config.get("dask_tmp_dir", "./")
        dask.config.set({'temporary-directory': dask_tmp_dir})
        # Initialize dask
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
        client = Client(cluster)
        results = []
        for ifile in range(len(datafiles)):
            print(datafiles[ifile])
            result = dask.delayed(work_for_time_loop)(datafiles[ifile], track_dict, map_info, plot_info, config)
            results.append(result)

        # Trigger dask computation
        final_result = dask.compute(*results)
    
    else:
        sys.exit('Valid parallelization flag not provided')
