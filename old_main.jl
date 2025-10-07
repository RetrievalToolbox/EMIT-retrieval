using ArchGDAL; const AG = ArchGDAL
using ArgParse
using CSV
using DataFrames
using Dates
using DistributedData
using HDF5
using Interpolations
using LinearAlgebra
using LinRegOutliers
using NCDatasets
using Polynomials
using ProgressMeter
using Statistics
using StatsBase
using Unitful

using RetrievalToolbox; const RE = RetrievalToolbox


include("args.jl")
include("helpers.jl")
include("forward_model.jl")

function main(root_waits_channel, wait_on_root_channel, ARGS_in)


    args = process_args(ARGS_in)

    # Define the root process to be the one with myid() == 1, but
    # any other number <= numprocs() should in theory work too..
    ROOT = 1

    #=
        Initial data read-ins and scattering to all processes
        This is wrapped into a sync because the following code must
        have all the needed symbols fetched at every remote worker.

        The ROOT process does all of the data read-ins, and then sends the data over
        to all others. Common data, where ALL processes need a copy of the data,
        are stored at ROOT, and all other processes then take it from ROOT.

        For data which varies from process to process (scene radiances etc.), they are
        `save_at` the specific process.

    =#

    nc_fname = args["L1B"]

    @sync if myid() == ROOT
        # Process ROOT reads from HD and then sends to others

        # Read the noise coefficients
        noise_csv = CSV.File(
            "data/emit_noise.txt", skipto=2,
            header=["wl", "a", "b", "c", "rmse"]
        )
        # Make it shareable
        save_at(ROOT, :noise_csv, noise_csv)

        # Open the NC L1b file
        nc_l1b = NCDataset(nc_fname)

        # Store the shape of the image (needed for output)
        global nc_shape = size(nc_l1b.group["location"]["lon"])

        # Read wavelength and send to others
        wavelengths = nc_l1b.group["sensor_band_parameters"]["wavelengths"].var[:]
        save_at(ROOT, :wavelengths, wavelengths)

        # Granule start time
        time_coverage_start = nc_l1b.attrib["time_coverage_start"]
        save_at(ROOT, :time_coverage_start, time_coverage_start)

        # Read ISRF FWHMs and send to others
        fwhms = nc_l1b.group["sensor_band_parameters"]["fwhm"].var[:]
        save_at(ROOT, :fwhms, fwhms)

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
        nc_lon = nc_l1b.group["location"]["lon"][:,:]
        nc_lat = nc_l1b.group["location"]["lat"][:,:]
        nc_alt = nc_l1b.group["location"]["elev"][:,:]

        nc_rad = nc_l1b["radiance"].var[:,:,:] # needs .var to ignore missing

        # Sub-set
        all_scene_idx = findall(
            (nc_lon .> lon_bounds[1]) .&
            (nc_lon .< lon_bounds[2]) .&
            (nc_lat .> lat_bounds[1]) .&
            (nc_lat .< lat_bounds[2])
            )

        # Split up the subset so every process gets something to do
        # (global because this is needed later on by ROOT to collect results)
        global all_proc_scene_idx = distribute_work(all_scene_idx)

        # Save each subset at other processes
        for p in 1:nprocs()
            this_idx = all_proc_scene_idx[p]

            save_at(p, :my_scene_idx, all_proc_scene_idx[p])
            save_at(p, :my_scene_lons, nc_lon[this_idx])
            save_at(p, :my_scene_lats, nc_lat[this_idx])
            save_at(p, :my_scene_alts, nc_alt[this_idx])
            save_at(p, :my_scene_rads, nc_rad[:, this_idx])

        end

        # Close the L1B file
        close(nc_l1b)

        # Create the solar model
        solar_model = RE.TSISSolarModel(
            "data/hybrid_reference_spectrum_p005nm_resolution_c2022-11-30_with_unc.nc",
            spectral_unit=:Wavelength
        )
        save_at(ROOT, :solar_model, solar_model)

        # Create the spectroscopy objects
        ABSCO_CH4 = RE.load_ABSCOAER_spectroscopy(
            "data/CH4_04000-05250_v0.0_init.nc";
            spectral_unit=:Wavelength, distributed=true
        )
        save_at(ROOT, :ABSCO_CH4, ABSCO_CH4)

        #= Add this later for CO2 retrievals
        ABSCO_CO2 = RE.load_ABSCOAER_spectroscopy(
            "data/CO2_04000-05250_v0.0_init.nc";
            spectral_unit=:Wavelength, distributed=true
        )
        save_at(ROOT, :ABSCO_CO2, ABSCO_CO2)
        =#

        ABSCO_H2O = RE.load_ABSCOAER_spectroscopy(
            "data/H2O_04000-05250_v0.0_init.nc";
            spectral_unit=:Wavelength, distributed=true
        )
        save_at(ROOT, :ABSCO_H2O, ABSCO_H2O)
    end

    # This barrier is needed for the next chunk where each process grabs the values
    # from itself that ROOT must put there first.
    wait_on_ROOT(wait_on_root_channel, ROOT)
    ROOT_waits(root_waits_channel, ROOT)
    println("Synchronization point after ROOT prepares data.")

    # Resolve process-issued data previously sent via by "save_at"
    my_scene_idx = get_val_from(myid(), :my_scene_idx)
    my_scene_lons = get_val_from(myid(), :my_scene_lons)
    my_scene_lats = get_val_from(myid(), :my_scene_lats)
    my_scene_alts = get_val_from(myid(), :my_scene_alts)
    my_scene_rads = get_val_from(myid(), :my_scene_rads)

    # Take data that was saved by root process
    noise_csv = get_val_from(ROOT, :noise_csv)
    wavelengths = get_val_from(ROOT, :wavelengths)
    time_coverage_start = get_val_from(ROOT, :time_coverage_start)
    fwhms = get_val_from(ROOT, :fwhms)

    solar_model = get_val_from(ROOT, :solar_model)

    ABSCO_CH4 = get_val_from(ROOT, :ABSCO_CH4)
    #ABSCO_CO2 = get_val_from(ROOT, :ABSCO_CO2) # add this later for CO2 retrievals
    ABSCO_H2O = get_val_from(ROOT, :ABSCO_H2O)

    # Let all procs catch up (need to have all data moved across before we can move on)
    wait_on_ROOT(wait_on_root_channel, ROOT)
    ROOT_waits(root_waits_channel, ROOT)
    println("Synchronization point after retrieving data.")

    #=
        Single-process part
        ===================

        From here on, the rest of the program can continue in single-process fashion;
        each process now has all the data needed to process the dataset it was given.

    =#

    #=
        Set up "global" definitions, derived from the data we have in hand
    =#
    Npix = length(wavelengths) # Number of bands

    # Produce linear noise interpolation for each coeff a,b,c
    noise_itps = Dict(
        "a" => linear_interpolation(noise_csv.wl, noise_csv.a, extrapolation_bc=Line()),
        "b" => linear_interpolation(noise_csv.wl, noise_csv.b, extrapolation_bc=Line()),
        "c" => linear_interpolation(noise_csv.wl, noise_csv.c, extrapolation_bc=Line()),
    )

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

    # Add more later if you want to, in case you want multi-band retrievals..



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
        1.0
    )

    # Retrieve a polynomial for the Lambertian surface albedo
    sv_surf = RE.SurfaceAlbedoPolynomialSVE[]
    surf_order = 3

    for (win_name, swin) in window_dict
        for o in 0:surf_order

            if o == 0
                fg = 0.25
            else
                fg = 0.0
            end

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
        sv_surf... # Expand list
    ])


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

    # EMIT noise model
    function noise_model(rad, a, b, c)
        return @. abs(sqrt(rad + b) * a + c)
    end

    # Function to calculate pressure from elevation (valid only for this model atmosphere)
    # (we want this to obtain the surface pressure given the L1B elevation)
    itp_logp_from_alt = linear_interpolation(
        reverse(atm.altitude_levels),
        log10.(reverse(atm.met_pressure_levels)),
        extrapolation_bc=Line()
    )

    # Function to calculate altitude from pressure
    # (we want this to produce the column-integrated CH4)
    # Note that we don't reverse here since the knots need to be in ascending order
    # and pressure levels increase when going towards the surface.
    itp_alt_from_logp = linear_interpolation(
        log10.(atm.met_pressure_levels),
        atm.altitude_levels,
        extrapolation_bc=Line()
    )

    #=
        Scene loop
        ==========
    =#

    # Before going into the scene loop, let's establish the solar strength for each
    # spectral window, so we can calculate a good initial guess for the surface albedo.
    solar_strength_guess = Dict{RE.SpectralWindow, Float64}()

    for (swin, rt) in buf.rt
        solar_idx = searchsortedfirst.(Ref(rt.solar_model.ww), swin.ww_grid[:] / 1000.0)
        solar_strength_guess[swin] = maximum(rt.solar_model.irradiance[solar_idx])
    end

    # ROOT will likely be the first one to reach this point, so let us make sure we wait
    # until all other processes catch up.
    ROOT_waits(root_waits_channel, ROOT)
    wait_on_ROOT(wait_on_root_channel, ROOT)

    my_range = StepRange(1, 1, length(my_scene_idx))

    println("$(myid()) is starting retrievals N = $(length(my_range))")
    flush(stdout)

    # result arrays - we want to store these
    CH4_result = zeros(Union{Missing, Float32}, length(my_range))
    albedo_result = zeros(Union{Missing, Float32}, length(my_range))
    chi2_result = zeros(Union{Missing, Float32}, length(my_range))
    maxrad_result = zeros(Union{Missing, Float32}, length(my_range))
    psurf_result = zeros(Union{Missing, Float32}, length(my_range))

    #for idx in my_range
    @showprogress dt=5 showspeed=true for idx in my_range

        CH4_result[idx] = missing
        albedo_result[idx] = missing
        chi2_result[idx] = missing
        maxrad_result[idx] = missing
        psurf_result[idx] = missing

        #=
        if (mod(idx, 50) == 0) # Show progress after every 50 scenes..
            # Print status
            println("$(idx) / $(length(my_range))")
            flush(stdout)
        end
        =#
        this_meas = my_scene_rads[:,idx] # Full Npix radiances

        this_noise = noise_model(
            this_meas,
            noise_itps["a"].(wavelengths),
            noise_itps["b"].(wavelengths),
            noise_itps["c"].(wavelengths),
        )

        solver = RE.IMAPSolver(
            forward_model!,
            state_vector,
            Diagonal(RE.get_prior_covariance(state_vector)), # Prior covariance matrix - just use diagonal
            10, # number of iterations
            3.0, # dsigma scale
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

            for (sve_idx, sve) in RE.StateVectorIterator( # loop through all albedo SVEs
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

        # Calculate new altitude and surface pressure
        new_altitude = my_scene_alts[idx]
        new_psurf = 10^(itp_logp_from_alt(new_altitude))
        new_plevels = generate_plevels(new_psurf * buf.scene.atmosphere.pressure_unit)

        # Move new retrieval grid into buffer
        RE.ingest!(buf.scene.atmosphere, :pressure_levels, new_plevels)

        # Calculate layer quantities from level quantities
        RE.calculate_layers!(buf.scene.atmosphere)

        # Set the scene location
        loc = RE.EarthLocation(
            my_scene_lons[idx],
            my_scene_lats[idx],
            new_altitude,
            u"m"
        )

        buf.scene.location = loc

        # Calculate solar angles from the location and time
        RE.update_solar_angles!(buf.scene)

        # Set the gas profiles back to their original prior state
        #gas_co2.vmr_levels[:] .= co2_prior
        gas_ch4.vmr_levels[:] .= ch4_prior
        gas_h2o.vmr_levels[:] .= h2o_prior

        iter_result = true

        # ITERATE!
        while !(RE.check_convergence(solver)) # Loop until converged

            # Evaluate and produce next iteration state vector!
            iter_result = RE.next_iteration!(solver; fm_kwargs)

            # Skip if bad results occur
            if !iter_result
                @info "Bad iteration at $(idx). Skipping."
                break
            end

            # Skip bad if NaNs in radiances etc.
            if !RE.check_solver_validity(solver)
                @info "Invalid solver at $(idx). Skipping."
                break
            end

            # Break if iterations reached limit.
            if RE.get_iteration_count(solver) > solver.max_iterations
                break
            end
        end

        # For a bad iteration, just skip to the next scene
        if !iter_result
            continue
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

        # Store the surface albedo and other data
        albedo_result[idx] = RE.get_current_value(sv_surf[1])
        chi2_result[idx] = collect(values(RE.calculate_chi2(solver)))[1]
        maxrad_result[idx] = maximum(RE.get_modeled(solver))
        psurf_result[idx] = buf.scene.atmosphere.pressure_levels[end]

        CH4_result[idx] = RE.calculate_xgas(buf.scene.atmosphere)["CH4"] |> u"ppb" |> ustrip
        #=
        # Store the CH4 as column-integrated value in units of ppm * m!
        CH4_result[idx] = 0.0

        for lay in 1:buf.scene.atmosphere.N_layer

            # Get altitudes for (retrieval grid) layer boundaries
            a1 = itp_alt_from_logp(log10(atm.pressure_levels[lay])) * atm.altitude_unit |> u"m"
            a2 = itp_alt_from_logp(log10(atm.pressure_levels[lay+1])) * atm.altitude_unit |> u"m"

            # Get mixing ratios in ppm (layer average from levels)
            m1 = gas_ch4.vmr_levels[lay] * gas_ch4.vmr_unit |> u"ppm"
            m2 = gas_ch4.vmr_levels[lay+1] * gas_ch4.vmr_unit |> u"ppm"

            # Add layer contribution to total
            # [(Δ altitude) * (layer mean VMR)] in ppm m
            CH4_result[idx] += (a1 - a2) * 0.5 * (m2 + m1) |> u"ppm * m" |> ustrip
        end
        =#
    end

    # After the retrievals are done, ROOT must wait here for all other processes to
    # gather the retrieval results.
    ROOT_waits(root_waits_channel, ROOT)
    wait_on_ROOT(wait_on_root_channel, ROOT)

    #=
        Gather results!
        ===============

        All others => ROOT
    =#

    # Save the results on each process
    save_at(myid(), :CH4_result, CH4_result)
    save_at(myid(), :albedo_result, albedo_result)
    save_at(myid(), :chi2_result, chi2_result)
    save_at(myid(), :maxrad_result, maxrad_result)
    save_at(myid(), :psurf_result, psurf_result)

    # ROOT must wait here on all other procs to finish saving the result.
    ROOT_waits(root_waits_channel, ROOT)

    if myid() == ROOT

        # Allocate big arrays for full results
        CH4_ENH = zeros(Union{Missing, Float32}, nc_shape...)
        CH4_ENH[:,:] .= missing

        ALBEDO = zeros(Union{Missing, Float32}, nc_shape...)
        ALBEDO[:,:] .= missing

        CHI2 = zeros(Union{Missing, Float32}, nc_shape...)
        CHI2[:,:] .= missing

        MAXRAD = zeros(Union{Missing, Float32}, nc_shape...)
        MAXRAD[:,:] .= missing

        PSURF = zeros(Union{Missing, Float32}, nc_shape...)
        PSURF[:,:] .= missing

        # Gather each process-based partial result
        for p in 1:nprocs()
            @info "Obtaining results from process $(p)"

            CH4_ENH[all_proc_scene_idx[p]] .= get_val_from(p, :CH4_result)
            ALBEDO[all_proc_scene_idx[p]] .= get_val_from(p, :albedo_result)
            CHI2[all_proc_scene_idx[p]] .= get_val_from(p, :chi2_result)
            MAXRAD[all_proc_scene_idx[p]] .= get_val_from(p, :maxrad_result)
            PSURF[all_proc_scene_idx[p]] .= get_val_from(p, :psurf_result)
        end

        # Take the median out
        #med = median(skipmissing(CH4_ENH))
        #CH4_ENH[:,:] .-= med

        # "Correct for surface"
        #df = DataFrame("ch4" => vec(CH4_ENH), "albedo" => vec(ALBEDO)) |> dropmissing
        #reg = createRegressionSetting(@formula(ch4 ~ albedo), df) # Set up createRegressionSetting
        #reg_result = lts(reg) # Fit

        #predict_ch4_enh = @. ALBEDO * reg_result["betas"][2] + reg_result["betas"][1]
        #CH4_ENH[:,:] .-= predict_ch4_enh # subtract surface effect



        #=
            CH4_ENH is in the shape of the lon/lat grid in the L1B file, but we should be
            saving the data accroding to the glt_x, glt_y
        =#

        #nc = NCDataset(nc_fname, "r")

        #glt_x = nc.group["location"]["glt_x"][:,:]
        #glt_y = nc.group["location"]["glt_y"][:,:]
        #=
        # Allocate result output
        output = zeros(size(glt_x)...)
        for i in axes(glt_x, 1), j in axes(glt_x, 2)

            output[i,j] = -9999.0

            if (glt_x[i,j] != 0)

                val = CH4_ENH[glt_x[i,j], glt_y[i,j]]

                if !ismissing(val)
                    output[i,j] = val # Save to output if not missing
                end
            end

        end

        # Set NaNs to nodatavalue
        output[isnan.(output)] .= -9999.0
        =#

        #Save out
        h5 = h5open(args["output"], "w")
        h5["ch4"] = replace(CH4_ENH, missing => NaN)
        h5["albedo"] = replace(ALBEDO, missing => NaN)
        h5["chi2"] = replace(CHI2, missing => NaN)
        h5["maxrad"] = replace(MAXRAD, missing => NaN)
        h5["psurf"] = replace(PSURF, missing => NaN)
        close(h5)

        # Save GeoTIFF (put this in new module)
        #=
        width = size(output, 1)
        height = size(output, 2)
        AG.create("test.tiff", driver=AG.getdriver("GTiff"),
            width=width, height=height, nbands=1, dtype=Float32) do gtiff

            ds = AG.write!(gtiff, output, 1)
            AG.setgeotransform!(gtiff, nc.attrib["geotransform"])
            AG.setproj!(gtiff, nc.attrib["spatial_ref"])

            band = AG.getband(ds, 1)
            AG.setnodatavalue!(band, -9999.0)

        end
        =#
        #close(nc)

    end

end