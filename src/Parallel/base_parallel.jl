function automatic_parallel_type()
    MPI.REFCOUNT[] == -1 && MPI.Init()
    return ParallelMPI();
end

function _iterator_blocks(interval, rank, n_par)
    rank         = rank -1
    n_min, extra = divrem(length(interval), n_par)
    iter_length  = n_min + (rank < extra ? 1 : 0)
    iter_start   = n_min * rank + min(rank, extra) + 1
    iter_end     = iter_start + iter_length - 1

    return iter_start:iter_end
end


function num_workers end

function worker_local_seed end

function worker_block end
