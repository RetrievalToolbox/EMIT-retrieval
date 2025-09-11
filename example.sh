#!/bin/bash

NUM_PROCS=9
echo "Spawning with ${NUM_PROCS} additional processes."

if (( NUM_PROCS == 0 )); then

    julia --project=./ run.jl \
        --L1B L1B/EMIT_L1B_RAD_001_20230612T162103_2316311_006.nc \
        --output test.h5 \
        --lon_bounds -111.3,-111.03 \
        --lat_bounds 41.2,41.5


else

    julia --project=./ -p ${NUM_PROCS} run.jl \
        --L1B L1B/EMIT_L1B_RAD_001_20230612T162103_2316311_006.nc \
        --output test.h5 \
        --lon_bounds -111.3,-111.03 \
        --lat_bounds 41.2,41.5

fi


# Small box with small plume
#lat_bounds 41.24,41.265
#lon_bounds -111.10,-111.06

# Slightly larger box with small plume
#lat_bounds 41.2375,41.280
#lon_bounds -111.1975,-111.085

# Larger box with larger plume
#lat_bounds 41.25,41.50
#lon_bounds -111.30,-111.10

# Even larger box containing both plumes
#lat_bounds 41.2,41.5
#lon_bounds -111.3,-111.03
