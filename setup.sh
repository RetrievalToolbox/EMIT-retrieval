#!/bin/bash

# A script to install this EMIT retrieval demo on an EC2-type machine (but likely to work
# on many other machines).

# Currently, we see the example working well in this version
julia_version=1.11.7

echo "### Installing Julia via juliaup"

juliaup -V
if [ $? -ne 0 ]; then
    echo "JuliaUp not found - installing"
    curl -fsSL https://install.julialang.org | sh -s --default-channel ${julia_version}
else
    echo "JuliaUp found - installing ${julia_version}."
    juliaup add ${julia_version}
fi

# Install Julia packages
julia +${julia_version} --project="./" -e 'using Pkg; Pkg.instantiate();'

# Install gdown to get Google Drive-stored data
echo "### Installing gdown into a venv"
python3 -m venv .venv
source .venv/bin/activate
pip install gdown

# Download needed additional files
./download_additional.sh
