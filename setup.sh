#!/bin/bash

# A script to install this EMIT retrieval demo on an EC2-type machine (but likely to work
# on many other machines).

# Currently, we see the example working well in this version
julia_version=1.11.7

juliaup -V # Check if JuliaUp is available at all
if [ $? -ne 0 ]; then
    echo "### JuliaUp not found - installing"
    curl -fsSL https://install.julialang.org | sh -s -- --default-channel ${julia_version}
    echo "### Julia was installed. Please re-start or re-source your shell!"
    exit
fi


# Check if the correct version is available

if `juliaup status | grep -q ${julia_version}`; then
    echo "### Correct Julia version ${julia_version} found!"
else
    echo "### JuliaUp found, but need a specific version - installing ${julia_version}."
    juliaup add ${julia_version}
fi

# Install Julia packages
echo "### Installing required Julia packages."
julia +${julia_version} --project="./" -e 'using Pkg; Pkg.instantiate();'

# Install gdown to get Google Drive-stored data
echo "### Installing gdown into a venv"
python3 -m venv .venv
source .venv/bin/activate
pip install gdown

# Download needed additional files
./download_additional.sh
