using ArchGDAL; const AG = ArchGDAL

using ArgParse
using HDF5
using NCDatasets
using ProgressMeter

function main()


    args_settings = ArgParseSettings()

    @add_arg_table! args_settings begin

        "--L1"
            help = "L1 NetCDF file"
            arg_type = String
            required = true

        "--L2"
            help = "L2 HDF5 file"
            arg_type = String
            required = true

        "--out"
            help = "Output file name"
            arg_type = String
            required = true

        end

        args = parse_args(ARGS, args_settings)


        nc_l1b = NCDataset(args["L1"])
        h5_l2 = h5open(args["L2"])

        # Get the mappings from L1b elements to "pixels"
        glt_x = nc_l1b.group["location"]["glt_x"][:,:]
        glt_y = nc_l1b.group["location"]["glt_y"][:,:]


        # Allocate output array
        output = zeros(size(glt_x)...)

        # Get the CH4 results (and others)
        XCH4 = h5_l2["XCH4"][:,:]

        # Filtering
        CHI2 = h5_l2["CHI2"][:,:]

        filter_good = @. (CHI2 > 0.1) & (CHI2 < 10) & (XCH4 > 1000)
        XCH4[(!).(filter_good)] .= NaN



        @showprogress for i in axes(glt_x, 1), j in axes(glt_x, 2)

            output[i,j] = -9999.0

            if (glt_x[i,j] != 0)

                val = XCH4[glt_x[i,j], glt_y[i,j]]

                if isfinite(val)
                    output[i,j] = val # Save to output if not missing
                end
            end

        end

        width = size(output, 1)
        height = size(output, 2)

        AG.create(args["out"], driver=AG.getdriver("GTiff"),
            width=width, height=height, nbands=1, dtype=Float32) do gtiff

            ds = AG.write!(gtiff, output, 1)
            AG.setgeotransform!(gtiff, nc_l1b.attrib["geotransform"])
            AG.setproj!(gtiff, nc_l1b.attrib["spatial_ref"])

            band = AG.getband(ds, 1)
            AG.setnodatavalue!(band, -9999.0)

        end


end

main()