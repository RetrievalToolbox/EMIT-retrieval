#!/bin/bash

julia_version=1.11.7
NUM_PROCS=${1}

# Define the L1B filename here!
l1b_fname=L1B/EMIT_L1B_RAD_demo.nc

if (( NUM_PROCS == 0 )); then

    julia +${julia_version} --project=./ main.jl \
        --L1B ${l1b_fname} \
        --output demo_results.h5 \

else

    echo "Spawning with ${NUM_PROCS} additional processes."

    julia +${julia_version} --project=./ -p ${NUM_PROCS} main.jl \
        --output demo_results.h5 \
        --L1B ${l1b_fname} \

fi
echo "Turning into GeoTIFF.."

julia +${julia_version} --project=./ produce_geotiff.jl \
    --L1 L1B/EMIT_L1B_RAD_demo.nc \
    --L2 demo_results.h5 \
    --out demo.tiff

# Small box with small plume
#--lon_bounds -111.10,-111.06 \
#--lat_bounds 41.24,41.265

# Slightly larger box with small plume
#--lon_bounds -111.1975,-111.085 \
#--lat_bounds 41.2375,41.280

# Larger box with larger plume
#--lon_bounds -111.30,-111.10 \
#--lat_bounds 41.25,41.50

# Even larger box containing both plumes
#--lon_bounds -111.3,-111.03 \
#--lat_bounds 41.2,41.5

# Part of larger plume with dark patch underneath
#--lon_bounds -111.215,-111.112 \
#--lat_bounds 41.35,41.45425
