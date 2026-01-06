#!/bin/bash
###############################################################################################
# This script demonstrates running cold pool tracking from PINACLES
# To run this demo script:
# 1. Modify the dir_demo to a directory on your computer to download the sample data
# 2. Run the script: bash demo_mrms_mcstracking.sh
#
# By default the demo config uses 128 processors for parallel processing.
###############################################################################################

# Specify start/end datetime
start_date='2000-03-21T00' #'2000-03-01T00' #'1970-02-11T00' #'2000-01-15T00' #'2000-01-26T00' #
end_date='2000-03-29T00' #'2000-04-17T00' #'1970-02-16T00' #'2000-02-08T00' #'2000-01-28T00' #
# Plotting map domain (lonmin lonmax latmin latmax)
map_extent='0. 600. 0. 600.'  # (xmin xmax ymin ymax)
run_parallel=1

# Specify directory for the demo data
dir_demo='/pscratch/sd/p/paccini/temp/tracking_coldpool_test/tracking_coldpool_match_pr_600_3km_robust_10min_Pfixed_v8_2/'
quicklook_dir=${dir_demo}'/quicklooks_coldpool_tracks_v3/'
animation_dir=${dir_demo}'/animations_v3/'
animation_filename=${animation_dir}coldpool_tracking_${start_date}_${end_date}.mp4

# Make quicklook & animation directories
mkdir -p ${quicklook_dir}
mkdir -p ${animation_dir}

# Example config file name
config_file='config_coldpool_buoyancy.yml'

# Activate PyFLEXTRKR conda environment
# echo 'Activating PyFLEXTRKR environment ...'
# source activate pyflex

# Run tracking
echo 'Running PyFLEXTRKR cold pool tracking ...'
python ../runscripts/run_generic_tracking.py ${config_file}
echo 'Cold pool tracking is done.'

# # Make quicklook plots for cold pools
echo 'Making cold pool quicklook plots ...'
python ../Analysis/plot_subset_coldpool_tracks_v2.py -s ${start_date} -e ${end_date} -c ${config_file} \
    -p ${run_parallel} --output ${quicklook_dir} \
    --extent "${map_extent}" --subset 0
echo 'View cold pool quicklook plots here: '${quicklook_dir}

# Make animation using ffmpeg
# Animation settings
vfscale='1200:-1'
framerate=4
echo 'Making cold pool tracking animations from quicklook plots ...'
# ffmpeg -framerate ${framerate} -pattern_type glob -i ${quicklook_dir}'*.png' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p \
ffmpeg -framerate ${framerate} -pattern_type glob -i ${quicklook_dir}'*.png' -c:v libx264 -crf 20 -pix_fmt yuv420p \
    -y ${animation_filename}
echo 'View cold pool animation here: '${animation_filename}

echo 'Cold pool demo completed!'