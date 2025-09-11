"""
Splits a vector into near-equal chunks. If the vector is shorter than the number of
workers, then empty sub-lists will be returned.
"""
function distribute_work(arr, nsub = nprocs())

    chunk_size = div(length(arr), nsub)
    remainder = length(arr) % nsub

    return [arr[i*chunk_size + 1 + min(i, remainder):(i+1)*chunk_size + min(i+1, remainder)]
            for i in 0:nsub-1]
end

"""
Synchronizes all workers to ROOT - makes sure the program waits here for ALL processes to catch
up before moving on.
"""
function wait_on_ROOT(barrier_channel, ROOT::Int=1)
    # Root process behavior
    if myid() == ROOT
        # Signal that root has arrived at the barrier
        N = nprocs()
        for x in 2:N
            @debug "[wait on ROOT] $(myid()) puts token $(x) into channel."
            put!(barrier_channel, true)
        end
    # Worker process behavior
    else
        # Wait for signal from root
        take!(barrier_channel)
        @debug "[wait on ROOT] $(myid()) takes token from channel."
    end
end


function ROOT_waits(barrier_channel, ROOT::Int=1)
    # Root process behavior
    if myid() == ROOT
        N = nprocs()
        for x in 2:N
            take!(barrier_channel)
            @debug "[ROOT waits] $(myid()) takes token $(x) from channel."
        end
    # Worker process behavior
    else
        @debug "[ROOT waits] $(myid()) puts token into channel."
        put!(barrier_channel, true)
    end
end