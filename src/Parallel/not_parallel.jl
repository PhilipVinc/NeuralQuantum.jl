parallel_execution_cache(::NotParallel) = NotParallel()

workers_mean!(target, data, par_type::NotParallel) =
    mean!(target, data)

workers_mean!(target, par_type::NotParallel) = target

workers_mean(data, par_type::NotParallel) = mean(data)

workers_sum!(target, par_type::NotParallel) = target
workers_sum!(target, data, par_type::NotParallel) = target .= data

num_workers(::NotParallel) = 1
worker_local_seed(seed, ::NotParallel) = seed
worker_block(iter, ::NotParallel) = iter
my_block(stuff, ::NotParallel) = stuff[1]

worker_allgatherv!(data, intervals, ::NotParallel) =
    data
