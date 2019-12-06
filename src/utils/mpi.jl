
function mean!(target, data, par_type::Parallel_MPI)
    mean!(target, data)
    MPI.Allreduce!(MPI.IN_PLACE, target, MPI.SUM, MPI.COMM_WORLD)
    target ./= MPI.Comm_size(MPI.COMM_WORLD)

    return target
end
