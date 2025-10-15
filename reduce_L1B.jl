using NCDatasets


nc = NCDataset("L1B/EMIT_L1B_RAD_demo.nc", "a")


lons = nc.group["location"]["lon"][:,:]
lats = nc.group["location"]["lat"][:,:]

mask = (
    (lons .>= -111.11429) .&
    (lons .<= -111.03702) .&
    (lats .>= 41.22825) .&
    (lats .<= 41.27453)
)

@info "Keeping $(sum(mask)) scenes."
mask_bad = (!).(mask)


# Set the following to missing
nc.group["location"]["lon"][mask_bad] .= -9999.0
nc.group["location"]["lat"][mask_bad] .= -9999.0
nc.group["location"]["elev"][mask_bad] .= -9999.0

nc["radiance"][:,mask_bad] .= -9999.0

# Close NC file
close(nc)