using Distributed

# Load modules everywhere
@everywhere begin

    using ArchGDAL; const AG = ArchGDAL
    using ArgParse
    using CSV
    using DataFrames
    using Dates
    using DistributedData
    using HDF5
    using Interpolations
    using LinearAlgebra
    using LoopVectorization
    using NCDatasets
    using Polynomials
    using ProgressMeter
    using SharedArrays
    using Statistics
    using StatsBase
    using Unitful

    using RetrievalToolbox; const RE = RetrievalToolbox

    include("args.jl")
    include("helpers.jl")
    include("forward_model.jl")

end


function main()


    # Process cmd line arguments
    args = process_args(ARGS)

    # Read the noise coefficients from TXT file
    noise_csv = CSV.File(
        "data/emit_noise.txt", skipto=2,
        header=["wl", "a", "b", "c", "rmse"]
    )


    #=
        Input file
        ==========

        Open the NC L1b file and read contents, distributing when needed
    =#

    nc_fname = args["L1B"]
    nc_l1b = NCDataset(nc_fname)

    # Store the shape of the image (needed for output)
    nc_shape = size(nc_l1b.group["location"]["lon"])

    # Read wavelength and send to others
    wavelengths = nc_l1b.group["sensor_band_parameters"]["wavelengths"].var[:] |> SharedArray
    @everywhere wavelengths = $wavelengths

    # Read ISRF FWHMs (needed later for creation of ISRFs)
    fwhms = nc_l1b.group["sensor_band_parameters"]["fwhm"].var[:]

    # Granule start time
    time_coverage_start = nc_l1b.attrib["time_coverage_start"]

    lon_bounds = [-180., 180.]
    if !isnothing(args["lon_bounds"])

        try
            lon_bounds = sort(parse.(Ref(Float64),
                split(args["lon_bounds"], ",")))
        catch
            @error "Could not parse longitude bounds!"
            exit(1)
        end
    end

    lat_bounds = [-90., 90.]
    if !isnothing(args["lat_bounds"])
        try
            lat_bounds = sort(parse.(Ref(Float64),
                split(args["lat_bounds"], ",")))
        catch
            @error "Could not parse latitude bounds!"
            exit(1)
        end
    end

    # Read all Lat/Lon/Alt
    nc_lon = replace(nc_l1b.group["location"]["lon"][:,:], missing => NaN) |> SharedArray
    @everywhere nc_lon = $nc_lon

    nc_lat = replace(nc_l1b.group["location"]["lat"][:,:], missing => NaN) |> SharedArray
    @everywhere nc_lat = $nc_lat

    nc_alt = replace(nc_l1b.group["location"]["elev"][:,:], missing => NaN) |> SharedArray
    @everywhere nc_alt = $nc_alt

    # needs .var to ignore missing
    nc_rad = replace(nc_l1b["radiance"][:,:,:], missing => NaN) |> SharedArray
    @everywhere nc_rad = $nc_rad

    # Sub-set (can be handled my main process only)
    all_scene_idx = findall(
        (nc_lon .> lon_bounds[1]) .&
        (nc_lon .< lon_bounds[2]) .&
        (nc_lat .> lat_bounds[1]) .&
        (nc_lat .< lat_bounds[2])
        )

    # Close the L1B file
    close(nc_l1b)

    # Create the solar model
    solar_model = RE.TSISSolarModel(
        "data/hybrid_reference_spectrum_p005nm_resolution_c2022-11-30_with_unc.nc",
        spectral_unit=:Wavelength
    )

    # Create the spectroscopy objects
    ABSCO_CH4 = RE.load_ABSCOAER_spectroscopy(
        "data/CH4_04000-05250_v0.0_init.nc";
        spectral_unit=:Wavelength, distributed=true
    )

    #= Add this later for CO2 retrievals
    ABSCO_CO2 = RE.load_ABSCOAER_spectroscopy(
        "data/CO2_04000-05250_v0.0_init.nc";
        spectral_unit=:Wavelength, distributed=true
    )
    =#

    ABSCO_H2O = RE.load_ABSCOAER_spectroscopy(
        "data/H2O_04000-05250_v0.0_init.nc";
        spectral_unit=:Wavelength, distributed=true
    )

    Npix = length(wavelengths) # Number of bands

    # Produce linear noise interpolation for each coeff a,b,c
    noise_itps = Dict(
        "a" => linear_interpolation(noise_csv.wl, noise_csv.a, extrapolation_bc=Line()),
        "b" => linear_interpolation(noise_csv.wl, noise_csv.b, extrapolation_bc=Line()),
        "c" => linear_interpolation(noise_csv.wl, noise_csv.c, extrapolation_bc=Line()),
    )
    @everywhere noise_itps = $noise_itps

    #=
        Retrieval windows
        =================
    =#
    window_dict = Dict{String, RE.SpectralWindow}()

    window_dict["CH4"] = RE.spectralwindow_from_ABSCO(
        "CH4",
        2150.0, # min
        2375.0, # max
        2300.0, # reference
        50.0, # buffer
        ABSCO_CH4, # Spectroscopy from which to get the wavelength grid from
        u"nm"
    );

   #=
        Dispersion objects
        ==================
    =#

    pixels = collect(1:Npix)
    # Make a polynomial fit to the wavelengths from the L1B file
    disp_fit = Polynomials.fit(pixels, wavelengths, 2) # order >2 does not seem to work!
    disp_coeffs = disp_fit.coeffs .|> Float64 # Get the coeffs

    dispersion_dict = Dict{RE.SpectralWindow, RE.SimplePolynomialDispersion}()
    for (wname, swin) in window_dict

        dispersion_dict[swin] = RE.SimplePolynomialDispersion(
            copy(disp_coeffs) * u"nm",
            1:Npix,
            swin # Reference this spectral window
        )

    end

    #=
        Instrument Spectral Response Function
        =====================================

        Build the ISRF table - we need a table since the ISRF changes with band/pixel

    =#


    Ndelta = 200
    wl_delta_unit = u"nm"
    wl_delta = zeros(Ndelta, Npix)
    rr = zeros(Ndelta, Npix)

    for i in 1:Npix

        σ = FWHM_to_sigma(fwhms[i]) # this is in nm

        wl_delta[:,i] = collect(LinRange(-7*σ, 7*σ, Ndelta))
        rr[:,i] = @. 1 / (σ * sqrt(2*pi)) * exp(-0.5 * ( - wl_delta[:,i])^2 / σ^2)
    end

    isrf = RE.TableISRF(
        wl_delta, # Δλ
        wl_delta_unit, # unit of Δλ
        rr # Relative response
    )

    # Create the dispersion => ISRF dictionary
    isrf_dict = Dict(disp => isrf for (swin, disp) in dispersion_dict)

    #=
        Gases
        =====

        Take representative profiles from the library (H2O, CH4, CO2 only)
    =#

    # Some user defined function to generate a pressure grid:
    function generate_plevels(psurf)
        return vcat(
            collect(LinRange(0.01u"hPa", 200.0u"hPa", 6)),
            collect(LinRange(300.0u"hPa", psurf, 5))
        )
    end

    plevels = generate_plevels(1000.0u"hPa")
    N_RT_lev = length(plevels)

    gases = RE.GasAbsorber[]

    gas_ch4 = RE.create_example_gas_profile("US-midwest-summer", "CH4", ABSCO_CH4, plevels)
    ch4_prior = copy(gas_ch4.vmr_levels)

    gas_h2o = RE.create_example_gas_profile("US-midwest-summer", "H2O", ABSCO_H2O, plevels)
    h2o_prior = copy(gas_h2o.vmr_levels)

    #gas_co2 = RE.create_example_gas_profile("US-midwest-summer", "CO2", ABSCO_CO2, plevels)
    #co2_prior = copy(gas_co2.vmr_levels)

    push!(gases, gas_ch4)
    push!(gases, gas_h2o)
    #push!(gases, gas_co2)

    @everywhere ch4_prior = $ch4_prior
    @everywhere h2o_prior = $h2o_prior

    #=
        Atmosphere
        ==========
    =#

    # Use `atm_orig` as the original reference atmosphere
    atm_orig = RE.create_example_atmosphere("US-midwest-summer", N_RT_lev; T=Float64);
    # Use `atm` as a working copy
    atm = deepcopy(atm_orig)
    N_MET_lev = atm.N_met_level

    # Ingest the retrieval grid..
    RE.ingest!(atm, :pressure_levels, plevels)

    # Make sure we have layer quantites everywhere
    # (they are calculated from level quantities)
    RE.calculate_layers!(atm)

    # Add gases to atmosphere
    push!(atm.atm_elements, gases...)


    #=
        State Vector
        ============
    =#

    sv_ch4_scaler = RE.GasLevelScalingFactorSVE(
        1,
        N_RT_lev,
        gas_ch4,
        Unitful.NoUnits,
        1.0,
        1.0,
        1.0
    )

    #= Uncomment for CO2
    sv_co2_scaler = RE.GasLevelScalingFactorSVE(
        1,
        N_RT_lev,
        gas_co2,
        Unitful.NoUnits,
        1.0,
        1.0,
        1.0e-4
    )
    =#

    sv_h2o_scaler = RE.GasLevelScalingFactorSVE(
        1,
        N_RT_lev,
        gas_h2o,
        Unitful.NoUnits,
        1.0,
        1.0,
        1.0e-1
    )

    # Retrieve a polynomial for the Lambertian surface albedo
    sv_surf = RE.SurfaceAlbedoPolynomialSVE[]
    @everywhere surf_order = 3

    for (win_name, swin) in window_dict
        for o in 0:surf_order

            o == 0 ? fg = 0.25 : fg = 0.0

            push!(sv_surf,
                RE.SurfaceAlbedoPolynomialSVE(
                    swin,
                    o,
                    u"nm",
                    fg,
                    fg,
                    1.0
                )
            )

        end
    end

    # Construct the state vector
    state_vector = RE.RetrievalStateVector([
        sv_ch4_scaler,
        sv_h2o_scaler,
        sv_surf..., # Expand list
    ])

    @everywhere state_vector = $state_vector


    #=
        Buffer
        ======
    =#

    N1 = 50_000 # Number of spectral points needed for monochromatic calculations, e.g. convolution
    N2 = Npix # Number of spectral points at instrument level needed

    my_type = Float64

    # Will contain outputs of ISRF application
    inst_buf = RE.InstrumentBuffer(
        zeros(my_type, N1),
        zeros(my_type, N1),
        zeros(my_type, N2),
    )

    # Buffer needed for optimal estimation linear algebra
    oe_buf = RE.OEBuffer(
        N2, length(state_vector), my_type
    )

    # Buffer needed for the monochromatic radiance calculations
    rt_buf = RE.ScalarRTBuffer(
        dispersion_dict, # Already a SpectralWindows -> Dispersion dictionary
        RE.ScalarRadiance(my_type, N2), # Hold the radiance - we use ScalarRadiance because we don't need polarization
        Dict(sve => RE.ScalarRadiance(my_type, N2) for sve in state_vector.state_vector_elements),
        Dict(swin => zeros(Int, 0) for swin in values(window_dict)), # Hold the detector indices
        u"mW * cm^-2 * sr^-1 * nm^-1" # Radiance units for the forward model
    )

    # Create the EarthAtmospherBuffer using this helper function rather than doing it manually
    buf = RE.EarthAtmosphereBuffer(
        state_vector, # The state vector
        values(window_dict) |> collect, # The spectral window (or a list of them)
        [(:Lambert, 5) for x in window_dict], # Surfaces
        atm.atm_elements, # All atmospheric elements
        Dict(swin => solar_model for swin in values(window_dict)), # Solar model dictionary (spectral window -> solar model)
        [:BeerLambert for swin in window_dict], # Use the speedy Beer-Lambert RT model
        RE.ScalarRadiance, # Use ScalarRadiance for high-res radiance calculations
        rt_buf,
        inst_buf,
        N_RT_lev, # The number of retrieval or RT pressure levels
        N_MET_lev, # The number of meteorological pressure levels, as given by the atmospheric inputs
        my_type # The chosen Float data type (e.g. Float16, Float32, Float64)
    )

    # Put our atmosphere in here
    buf.scene.atmosphere = atm

    # Set the time (roughly)
    buf.scene.time = DateTime(split(time_coverage_start, "+")[1]);

    # Set satellite observer
    # (the EMIT L1B files do not have satellite angles in them, so we just pretend it's
    #  all at nadir)
    buf.scene.observer = RE.SatelliteObserver(
        0.0, # zenith angle (let's assume it's close to nadir)
        0.0, # azimuth angle (does not matter here)
        [0., 0., 0.], # Satellite position
        [0., 0., 0.] # Satellite velocity
    )

    @info "Spawning buffers everywhere.."
    @everywhere buf = $buf

    #=
        Forward model definitions
    =#

    fm_kwargs = (
        buf=buf,
        inst_buf=inst_buf,
        oe_buf=oe_buf,
        rt_buf=rt_buf,
        dispersions=dispersion_dict,
        isrf_dict=isrf_dict,
        solar_doppler_factor=nothing, # Let RetrievalToolbox calculate the solar Doppler shift
        solar_distance=1.0
        );

    @everywhere fm_kwargs = $fm_kwargs

    # EMIT noise model
    @everywhere function noise_model!(noise, rad, a, b, c)
        @turbo for i in eachindex(noise)
            noise[i] = abs(a[i] * sqrt(rad[i] + b[i]) + c[i])
        end

        # Bump up negative noise to this (as per reference algorithm)
        noise[noise .<= 0] .= 1e-5

    end

    # Pre-allocate a vector to contain the noise, so we don't have to do it over and over
    @everywhere this_noise = zeros(size(nc_rad, 1))

    # Pre-calculate the a,b,c as function of wavelength!
    @everywhere noise_a = noise_itps["a"].(wavelengths)
    @everywhere noise_b = noise_itps["b"].(wavelengths)
    @everywhere noise_c = noise_itps["c"].(wavelengths)

    # Function to calculate pressure from elevation (valid only for this model atmosphere)
    # (we want this to obtain the surface pressure given the L1B elevation)
    @everywhere itp_logp_from_alt = linear_interpolation(
        reverse(buf.scene.atmosphere.altitude_levels),
        log10.(reverse(buf.scene.atmosphere.met_pressure_levels)),
        extrapolation_bc=Line()
    )

    # Function to calculate altitude from pressure
    # (we want this to produce the column-integrated CH4)
    # Note that we don't reverse here since the knots need to be in ascending order
    # and pressure levels increase when going towards the surface.
    @everywhere itp_alt_from_logp = linear_interpolation(
        log10.(buf.scene.atmosphere.met_pressure_levels),
        buf.scene.atmosphere.altitude_levels,
        extrapolation_bc=Line()
    )

    # Before going into the scene loop, let's establish the solar strength for each
    # spectral window, so we can calculate a good initial guess for the surface albedo.
    @everywhere solar_strength_guess = Dict{RE.SpectralWindow, Float64}()

    for (swin, rt) in buf.rt
        solar_idx = searchsortedfirst.(Ref(rt.solar_model.ww), swin.ww_grid[:] / 1000.0)
        solar_strength_guess[swin] = maximum(rt.solar_model.irradiance[solar_idx])
    end

    # Allocate output containers. The dict contains shared arrays and can be written into
    # by all processes.

    result_keys = [
        ("CONVERGED", Bool),
        ("CHI2", Float32),
        ("XCH4", Float32),
        ("CH4_SCALER", Float32),
        ("H2O_SCALER", Float32),
        ("CH4_SCALER_UCERT", Float32),
        ("H2O_SCALER_UCERT", Float32),
        ("XCH4_PRIOR", Float32),
        ("ITERATIONS", Int),
        ("SNR", Float32)
        ]

    for i in 0:surf_order
        push!(result_keys, ("ALBEDO_ORDER_$(i)", Float32))
    end

    result_container = Dict{String, SharedArray}()

    # Allocate and set all results to NaN by default (if float, -9999 otherwise)
    for (key, key_type) in result_keys
        result_container[key] = zeros(key_type, nc_shape) |> SharedArray

        if (key_type <: AbstractFloat)
            result_container[key][:,:] .= NaN
        end
        if (key_type <: Integer) & (key_type != Bool) # (Bool <: Integer sadly..)
            result_container[key][:,:] .= -9999
        end
        if (key_type == Bool)
            result_container[key][:,:] .= false
        end

    end

    @everywhere result_container = $result_container

    #=
        Scene loop
        ==========
    =#

    # Execute the retrievals in a @distributed loop.
    # EVERY quantity inside the loop must be available to all workers/processes

    @info "Processing $(length(all_scene_idx)) scenes."

    @showprogress showspeed=true dt=3 @distributed for idx in all_scene_idx

        # We must first take all the per-scene data from the various shared arrays.
        this_meas = @view nc_rad[:,idx] # Radiance / measurement
        this_lon = nc_lon[idx]
        this_lat = nc_lat[idx]
        this_alt = nc_alt[idx]

        # Adjust surface pressure for this scene to follow the terrain..
        this_psurf = 10^(itp_logp_from_alt(this_alt))

        # Move new retrieval grid into buffer
        new_plevels = generate_plevels(this_psurf * buf.scene.atmosphere.pressure_unit)
        RE.ingest!(buf.scene.atmosphere, :pressure_levels, new_plevels)

        # Calculate layer quantities from level quantities
        RE.calculate_layers!(buf.scene.atmosphere)

        # Set the scene location
        loc = RE.EarthLocation(
            this_lon,
            this_lat,
            this_alt,
            u"m"
        )
        buf.scene.location = loc

        # Calculate solar angles from the location and time
        # RE.update_solar_angles!(buf.scene)

        #gas_co2 = RE.get_gas_from_name(buf.scene.atmosphere, "CO2")
        gas_ch4 = RE.get_gas_from_name(buf.scene.atmosphere, "CH4")
        gas_h2o = RE.get_gas_from_name(buf.scene.atmosphere, "H2O")

        # Set the gas profiles back to their original prior state
        #gas_co2.vmr_levels[:] .= co2_prior
        gas_ch4.vmr_levels[:] .= ch4_prior
        gas_h2o.vmr_levels[:] .= h2o_prior

        result_container["XCH4_PRIOR"][idx] = (
            RE.calculate_xgas(buf.scene.atmosphere)["CH4"] |> u"ppb" |> ustrip
        )

        # Calculate the noise (in-place)!
        noise_model!(this_noise, this_meas,
            noise_itps["a"].(wavelengths),
            noise_itps["b"].(wavelengths),
            noise_itps["c"].(wavelengths),
        )

        # Create solver
        solver = RE.IMAPSolver(
            forward_model!,
            state_vector,
            Diagonal(RE.get_prior_covariance(state_vector)), # Prior covariance matrix - just use diagonal
            #10.0, # Smaller steps..
            20, # number of iterations
            0.5, # dsigma scale
            dispersion_dict,
            rt_buf.indices,
            rt_buf.radiance,
            rt_buf.jacobians,
            Dict(disp => this_meas for disp in values(dispersion_dict)), # the measurement (full spectrometer)
            Dict(disp => this_noise for disp in values(dispersion_dict)) # Noise (full spectrometer)
        )

        # Adjust surface reflectance first guess for every retrieval window
        for (swin, rt) in buf.rt

            # Find out any unit conversion factor between measured radiance (rt_buf.radiance_unit)
            # and the radiance we are using internally.
            rad_unit_fac = 1.0 * rt_buf.radiance_unit / rt.radiance_unit |> upreferred

            # Calculate apparent albedo from the measured radiances
            signal = percentile(RE.get_measured(solver, swin; view=true), 99)
            albedo_prior = pi * signal / (
                solar_strength_guess[swin] * cosd(buf.scene.solar_zenith)) * rad_unit_fac

            for (sve_idx, sve) in RE.StateVectorIterator( # lopp through all albedo SVEs
                state_vector, RE.SurfaceAlbedoPolynomialSVE)
                if sve.coefficient_order == 0
                    sve.first_guess = albedo_prior
                    sve.prior_value = albedo_prior
                end
            end
        end

        # Re-set the state vector to first guess values (empty the iterations)
        for sve in state_vector.state_vector_elements
            empty!(sve.iterations)
            push!(sve.iterations, sve.first_guess)
        end


        iter_result = true
        converged = false

        # ITERATE!
        while !(RE.check_convergence(solver)) # Loop until converged

            # Evaluate and produce next iteration state vector!
            try
                iter_result = RE.next_iteration!(solver; fm_kwargs)
            catch
                # Something bad happened..
                @info "ERROR during iteration."
                iter_result = false
                break
            end

            # Skip if bad results occur
            if !iter_result
                #@info "Bad iteration at $(idx). Skipping."
                break
            end

            # Skip bad if NaNs in radiances etc.
            if !RE.check_solver_validity(solver)
                #@info "Invalid solver at $(idx). Skipping."
                # in case iter_results returns a true beforehand..
                iter_result = false
                break
            end

            # Break if iterations reached limit.
            if RE.get_iteration_count(solver) > solver.max_iterations
                break
            end

            # Clamp gas scale factors
            for (sve_idx, sve) in RE.StateVectorIterator(
                state_vector, RE.GasLevelScalingFactorSVE)

                current = RE.get_current_value(sve)

                #if (current < 0.85)
                #    sve.iterations[end] = 0.85
                #end
                #if (current > 3.0) # Strong plumes might be 3x?
                #    sve.iterations[end] = 3.0
                #end

            end

        end

        # For a bad retrieval, just skip to the next scene
        if !iter_result
            continue
        end

        if RE.check_convergence(solver)
            converged = true
        end

        # Update the atmosphere object
        RE.atmosphere_statevector_update!(
            buf.scene.atmosphere,
            state_vector
        )

        # Update the atmosphere elements (this must be done separately)
        RE.atmosphere_element_statevector_update!(
            buf.scene.atmosphere.atm_elements,
            state_vector
        )

        # Do error analysis
        q = RE.calculate_OE_quantities(solver)

        if isnothing(q)
            # If we can't do error analysis, something probably went wrong..
            continue
        end

        # Put results into container
        result_container["CONVERGED"][idx] = converged
        result_container["SNR"][idx] = mean(RE.get_measured(solver) ./ RE.get_noise(solver))
        result_container["CHI2"][idx] = collect(values(RE.calculate_chi2(solver)))[1]
        result_container["XCH4"][idx] = RE.calculate_xgas(buf.scene.atmosphere)["CH4"] |> u"ppb" |> ustrip
        result_container["ITERATIONS"][idx] = RE.get_iteration_count(solver)

        for (sve_idx, sve) in RE.StateVectorIterator(
            state_vector, RE.SurfaceAlbedoPolynomialSVE)
            o = sve.coefficient_order

            result_container["ALBEDO_ORDER_$(o)"][idx] = RE.get_current_value(sve)
        end

        for (sve_idx, sve) in RE.StateVectorIterator(
            state_vector, RE.GasLevelScalingFactorSVE)

            gname = sve.gas.gas_name

            result_container["$(gname)_SCALER"][idx] = RE.get_current_value(sve)
            result_container["$(gname)_SCALER_UCERT"][idx] = q.SV_ucert[sve_idx]
        end
    end

    # Main process produces the output file
    h5open(args["output"], "w") do h5out

        for key in keys(result_container)
            @info "Writing out $(key)"
            h5out[key, compress=2] = result_container[key]
        end

    end



end


main()