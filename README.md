# EMIT greenhouse gas retrieval via RetrievalToolbox

This is a demonstration showing how a greenhouse gas retrieval for EMIT measurements can be implemented with the **RetrievalToolbox** algorithm tools.

The demo is split into two parts:

- An interactive notebook, `Demonstration.ipynb`, which shows a single-scene retrieval that can be run by users at their own pace, step by step. The notebook contains documentation to help users to understand what is all needed to create a CH4/CO2 retrieval from hyperspectral imagers, such as NASA's EMIT.

- A batch processing script, `example.sh`, which launches the retrieval for all scenes within the L1B file.


To run the example(s), some additional data is needed which currently resides on a public Google Drive folder. The files are automatically downloaded when the `setup.sh` script is run. These data include the following

1. Spectroscopy files that were generated with the [ReFRACtor ABSCO toolset](https://github.com/ReFRACtor/ABSCO)
2. The LASP TSIS solar model, which can also be downloaded [here](https://lasp.colorado.edu/lisird/data/tsis1_hsrs_p1nm)
3. An reduced EMIT L1B file, which has been stripped of most valid scenes, apart from a small section which contains a visible CH4 plume. The granule is originally `EMIT_L1B_RAD_001_20230612T162103_2316311_006.nc` and can be downloaded from [NASA's Earthdata catalogue](https://www.earthdata.nasa.gov/data/catalog/lpcloud-emitl1brad-001).

> [!WARNING]
> This is a demonstration only intended to show how **RetrievalToolbox** can be used to implement a gas retrieval for hyperspectral imaging instruments, such as NASA's EMIT. The retrieval algorithm demonstrated here is **not** a fully-tested application. The inferred methane and carbon dioxide concentrations are not accurate, no bias correction or validation against either ground-truths or other retrieval products has been done.


## Requirements

The retrieval algorithm and scripts are designed to work on a Unix-like system (Mac OS, Linux) and require only `bash`, a recent version of Python that includes `pip`, and the ability to download and run the `JuliaUp` installer via curl. Administrator privileges are not needed.


## Instructions

- First, clone the repository into a location of your choice with `git clone https://github.com/PeterSomkuti/EMIT-retrieval.git`
- Navigate into the `EMIT-retrieval` directory and run
    `./setup.sh`
  - This will download and install the most recent and stable Julia (the language in which EMIT-retrieval is written in) distribution, and then download all packages required to run the examples.
  - Further, a new Python virtual environment is created into which the [`gdown`](https://github.com/wkentaro/gdown) module is installed, which lets us download the required additional files.


### Running the interactive notebook

To run the interactive notebook paste following command into a terminal after navigating into the `EMIT-retrieval` directory that was cloned earlier:

`julia --project=./ -e "using IJulia; IJulia.notebook(dir=pwd())"`

Users might be prompted by `IJulia` to download a version of `JupyterLab` if it is not already installed. Once done, a new browser window will be opened with the classic "notebook" interface. From there simply open up the `Demonstration.ipynb` file and follow the instructions within.

For users who prefer the newer "JupyterLab" interface, run the following instead:

`julia --project=./ -e "using IJulia; IJulia.jupyterlab(dir=pwd())"`

Should difficulties arise in this step, please refer to [IJulia](https://github.com/JuliaLang/IJulia.jl) or [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/).


### Running the batch process

Navigate into the previously cloned `EMIT-retrieval` directory and start the batch processing via

`./example.sh`

This will run batch processing of ~8,000 scenes. The output will then be converted into a compliant GeoTIFF (`demo.tiff`) file that can be viewed with e.g. GIS applications. The retrieval application can utilize additional processes. On most modern machines users can add a number to the command which will then spawn additional processes that also partake in the processing of scenes. For example:

`./example.sh 7`

will cause 7 **additional** processes to be spawned, for a total of 8. In this multi-processing mode, also a progress bar will appear that informs the user of the progress every few seconds.

Note that the batch processing is setup to only retrieve CH4 from a single retrieval window. Once successful, the retrieval should reproduce the methane enhancement due to a plume as shown below. Note that there is no major post-processing in this example, thus surface features will imprint visibly on the XCH4 enhancement fields.

![Example methane plume, observed by EMIT at granule EMIT_L1B_RAD_001_20230612T162103_2316311_006](plume_example.png)

## References

- [NASA EMIT mission](https://earth.jpl.nasa.gov/emit/)
- [EMIT GHG retrieval repository](https://github.com/emit-sds/emit-ghg)
- [ReFRACtor ABSCO toolset](https://github.com/ReFRACtor/ABSCO)
- [RetrievalToolbox.jl](https://github.com/US-GHG-Center/RetrievalToolbox.jl)
- [RetrievalToolbox learning materials](https://petersomkuti.github.io/RetrievalToolbox-Tutorials/)