#!/bin/bash
###############################################################################################
# This script demonstrates running MCS tracking on GPM IMERG Tb + precipitation data
# To run this demo script:
# 1. Modify the dir_demo to a directory on your computer to download the sample data
# 2. Run the script: bash demo_mcs_imerg.sh
# 
# By default the demo config uses 4 processors for parallel processing, 
#    assuming most computers have at least 4 CPU cores. 
#    If your computer has more than 4 processors, you may modify 'nprocesses' 
#    in config_imerg_mcs_tbpf_example.yml to reduce the run time.
###############################################################################################

# Specify start/end datetime
start_date='2011-04-27T00' 
end_date='2011-04-30T00' 

# Plotting map domain (lonmin lonmax latmin latmax)
map_extent='-180. -105 -30 30'  # (xmin xmax ymin ymax)
run_parallel=1

# Specify directory for the demo data
# There are a total of 4 tests (e.g., test1, test2, test3, test4)
dir_demo='/pscratch/sd/p/paccini/temp/output_tracking/tracking_mcs_obs_nopbc_v4/'

# Example config file name
config_demo='/global/cfs/cdirs/wcm_code/lpaccini/PyFLEXTRKR-dev/config/config_imerg_mcs_tbpf_example_test.yml'


# Demo input data directory
dir_input='/pscratch/sd/p/paccini/temp/sample_data/'
quicklook_dir=${dir_demo}'/quicklooks_trackpaths/'
animation_dir=${dir_demo}'/animations/'
animation_filename=${animation_dir}mcs_tracking_${start_date}_${end_date}.mp4

# Make quicklook & animation directories
mkdir -p ${quicklook_dir}
mkdir -p ${animation_dir}

# Run tracking
echo 'Running PyFLEXTRKR ...'
python /global/cfs/cdirs/wcm_code/lpaccini/PyFLEXTRKR-dev/runscripts/run_mcs_tbpf_saag.py ${config_demo}
echo 'Tracking is done.'

# Make quicklook plots
echo 'Making quicklook plots ...'

# python /global/cfs/cdirs/wcm_code/lpaccini/PyFLEXTRKR-dev/Analysis/plot_subset_tbpf_mcs_tracks_demo.py -s ${start_date} -e ${end_date} \
#      -c ${config_demo} -o vertical -p 1 --figsize 10 8 --output ${quicklook_dir}
 
python /global/cfs/cdirs/wcm_code/lpaccini/PyFLEXTRKR-dev/Analysis/plot_subset_tbpf_tracks_nomap_obs_pbc.py -s ${start_date} -e ${end_date} \
     -c ${config_demo} -p 1 --figsize 10 8 --output ${quicklook_dir}  --extent "${map_extent}" --subset 0

    # --figbasename 'image' --figname_type 'sequence'
echo 'View quicklook plots here: '${quicklook_dir}

# Make animation using ffmpeg
vfscale='1200:-1'
framerate=2
echo 'Making animations from quicklook plots ...'
#ffmpeg -framerate 2 -pattern_type sequence -start_number 00001 -i ${quicklook_dir}'image%05d.png' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p \
ffmpeg -framerate ${framerate} -pattern_type glob -i ${quicklook_dir}'*.png' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p \
    -y ${animation_filename}
echo 'View animation here: '${animation_filename}

echo 'Demo completed!'
