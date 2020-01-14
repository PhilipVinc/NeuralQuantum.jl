mutable struct ThreadsCache{A, B, C}
    comm::A
    world_sz::B
    rank::C
end

function ThreadsCache(comm::TPI.Comm=TPI.Comm())
    return ThreadsCache(comm,
    TPI.Comm_size(comm),
    TPI.Comm_rank(comm))
end

function ThreadsCaches(n::Int)
    cms = TPI.Comms(n)
    return [ThreadsCache(c) for c=cms]
end

parallel_execution_cache(::ParallelThreaded) = ThreadsCache()
parallel_execution_cache(tc::ThreadsCache) = ThreadsCache(tc.comm)

num_workers(tc::ThreadsCache) = tc.world_sz

function workers_mean!(target, data, par_type::ThreadsCache)
    mean!(target, data)
    TPI.Allreduce!(target, target, +, par_type.comm)
    target ./= par_type.world_sz

    return target
end

function workers_mean!(target, par_type::ThreadsCache)
    TPI.Allreduce!(target, +, par_type.comm)
    target ./= par_type.world_sz

    return target
end

function workers_mean(data, par_type::ThreadsCache)
    data_m = mean(data)
    target = TPI.Allreduce(data_m, +, par_type.comm)

    return target/par_type.world_sz
end

function workers_sum!(target, par_type::ThreadsCache)
    TPI.Allreduce!(target, +, par_type.comm)
    return target
end

function all_bcast_from_root!(data, par_type::ThreadsCache)
    TPI.Bcast!(data, 1, par_type.comm)
    return data
end

function worker_local_seed(seed, par_cache::ThreadsCache)
    rng = Random.MersenneTwister(seed)
    local_seeds = rand(rng, eltype(seed), par_cache.world_sz)
    TPI.Bcast!(local_seeds, 1, par_cache.comm)
    return local_seeds[Threads.threadid()]
end

function worker_block(iter, par_cache::ThreadsCache)
    return _iterator_blocks(iter, Threads.threadid(), par_cache.world_sz)
end

my_block(stuff, par_cache::ThreadsCache) = stuff[Threads.threadid()]

function worker_allgatherv!(data, counts, par_cache::ThreadsCache)
    TPI.Allgatherv!(data, counts, par_cache.comm)
    return data
end
