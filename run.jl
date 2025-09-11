using Distributed

# These barrier channels are needed for distributed processing to make sure that
# worker processes do not run away from root.
root_waits_channel = RemoteChannel(() -> Channel{Bool}(nprocs()))
wait_on_root_channel = RemoteChannel(() -> Channel{Bool}(nprocs()))


# We have to make sure all workers have the same command line arguments before
# we enter main.jl

@everywhere include("main.jl")
@everywhere main($root_waits_channel, $wait_on_root_channel, $ARGS)