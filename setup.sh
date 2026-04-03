#!/bin/bash

# A script to install this EMIT retrieval demo on an EC2-type machine (but likely to work
# on many other machines).

juliaup -V # Check if JuliaUp is available at all
if [ $? -ne 0 ]; then
    echo "### JuliaUp not found - installing"
    curl -fsSL https://install.julialang.org
    echo "### Julia was installed. Please re-start or re-source your shell!"
    exit
fi

# Install Julia packages
echo "### Installing required Julia packages."
julia --project="./" -e 'using Pkg; Pkg.instantiate();'

# Install gdown to get Google Drive-stored data
if [ -f .venv/bin/gdown ]; then
    echo "### gdown already installed in venv"
else
    echo "### Installing gdown into a venv"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install gdown
fi

# Download needed additional files
./download_additional.sh
