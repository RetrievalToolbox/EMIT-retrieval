function process_args(ARGS_in)

    args_settings = ArgParseSettings()

    @add_arg_table! args_settings begin

        "--lon_bounds"
            help = "Longitude bounds, optional, separated by comma"
            arg_type = String
            required = false

        "--lat_bounds"
            help = "Latitude bounds, optional, separated by comma"
            arg_type = String
            required = false

        "--L1B"
            help = "Path to L1B grnaule"
            arg_type = String
            required = true

        "--output"
            help = "Path to output HDF5 file"
            arg_type = String
            required = true

    end

    return parse_args(ARGS_in, args_settings)

end