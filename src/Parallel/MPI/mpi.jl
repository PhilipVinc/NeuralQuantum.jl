struct MPIData{A,B,C}
    comm::A
    world_sz::B
    rank::C
end

function MPIData(comm=default_comm())
    return MPIData(comm, MPI.Comm_size(comm), MPI.Comm_rank(comm))
end

function default_comm()
    # Initialize MPI if not already done
    MPI.REFCOUNT[] == -1 && MPI.Init()
    return MPI.COMM_WORLD
end

parallel_execution_cache(::ParallelMPI) = MPIData()
num_workers(tc::MPIData) = tc.world_sz

## stuff
function workers_mean!(target, data, par_type::MPIData)
    mean!(target, data)
    MPI.Allreduce!(MPI.IN_PLACE, target, MPI.SUM, par_type.comm)
    target ./= par_type.world_sz

    return target
end

function workers_mean!(target, par_type::MPIData)
    MPI.Allreduce!(MPI.IN_PLACE, target, MPI.SUM, par_type.comm)
    target ./= par_type.world_sz

    return target
end

function workers_mean(data, par_type::MPIData)
    data_m = mean(data)
    target = MPI.Allreduce(data_m, MPI.SUM, par_type.comm)

    return target/par_type.world_sz
end

function workers_sum!(target, par_type::MPIData)
    MPI.Allreduce!(target, MPI.SUM, par_type.comm)
    return target
end

function all_bcast_from_root!(data, par_type::MPIData)
    MPI.Bcast!(data, 0, par_type.comm)
    return data
end

function worker_local_seed(seed, par_cache::MPIData)
    rng = Random.MersenneTwister(seed)
    local_seeds = rand(rng, eltype(seed), par_cache.world_sz)
    MPI.Bcast!(local_seeds, 0, par_cache.comm)
    rank = MPI.Comm_rank(par_cache.comm) +1
    return local_seeds[rank]
end

function worker_block(iter, par_cache::MPIData)
    return _iterator_blocks(iter, MPI.Comm_rank(par_cache.comm) + 1, par_cache.world_sz)
end

my_block(stuff, par_cache::MPIData) = stuff[MPI.Comm_rank(par_cache.comm) + 1]

function gather_data!(data, intervals, par_cache::MPIData)
    loc_len = 3
end

function worker_allgatherv!(data, counts, par_cache::MPIData)
    MPI.Allgatherv!(MPI.IN_PLACE, data, counts, par_cache.comm)
    return data
end
