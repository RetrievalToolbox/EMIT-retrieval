#!/bin/bash

source .venv/bin/activate
echo "Downloading required data..."

cd data/
# CH4 spectroscopy
gdown -c https://drive.google.com/uc?id=1Kvh_LuLa-lUne8-y8MVDc-2jg8TCVM9_
# CO2 spectroscopy
gdown -c https://drive.google.com/uc?id=1TbcGfdlgCnSFMzJ-gH7pulxyK-QCYEBb
# H2O spectroscopy
gdown -c https://drive.google.com/uc?id=1k0AgffwkCPfCuBLAws_OwsrQw5n13h6e
# Solar model
gdown -c https://drive.google.com/uc?id=1ENqe6VSD8yg2rXY1fFI7i2t8OwjTdqqp
cd ../

# Download the EMIT L1B file
cd L1B/
gdown -c https://drive.google.com/uc?id=1gjf2ZE0Qj9W3mR8knqDJrhs4uwUfXJZ_
cd ../