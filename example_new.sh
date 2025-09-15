#!/bin/bash

NUM_PROCS=${1}
echo "Spawning with ${NUM_PROCS} additional processes."

if (( NUM_PROCS == 0 )); then

    julia --project=./ new_way.jl \
        --L1B L1B/EMIT_L1B_RAD_001_20230612T162103_2316311_006.nc \
        --output test.h5 \
        --lon_bounds -111.10,-111.06 \
        --lat_bounds 41.24,41.265

else

    julia --project=./ -p ${NUM_PROCS} new_way.jl \
        --L1B L1B/EMIT_L1B_RAD_001_20230612T162103_2316311_006.nc \
        --output test.h5 \
        --lon_bounds -111.215,-111.112 \
        --lat_bounds 41.35,41.45425

fi


# Small box with small plume
#lon_bounds -111.10,-111.06
#lat_bounds 41.24,41.265

# Slightly larger box with small plume
#lon_bounds -111.1975,-111.085
#lat_bounds 41.2375,41.280

# Larger box with larger plume
#lon_bounds -111.30,-111.10
#lat_bounds 41.25,41.50

# Even larger box containing both plumes
#lon_bounds -111.3,-111.03
#lat_bounds 41.2,41.5
