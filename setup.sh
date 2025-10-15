#!/bin/bash

# A script to install this EMIT retrieval demo on an EC2-type machine (but likely to work
# on many other machines).

echo "### Installing Julia via juliaup"

juliaup -V
if [ $? -ne 0 ]; then
    echo "JuliaUp not found - installing"
    curl -fsSL https://install.julialang.org | sh
else
    echo "JuliaUp found - moving on."
fi

# Install Julia packages
julia --project="./" -e 'using Pkg; Pkg.instantiate();'

# Install gdown to get Google Drive-stored data
echo "### Installing gdown into a venv"
python3 -m venv .venv
source .venv/bin/activate
pip install gdown

# Download needed additional files
./download_additional.sh

