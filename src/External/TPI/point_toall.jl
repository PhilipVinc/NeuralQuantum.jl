"""
    TPI.Bcast!(data, root, comm)

Broadcasts the content of `data` on thread `root` to the other threads.
"""
function Bcast!(data::A, root, comm::Comm) where {T, A<:DenseArray{T}}
    sz = size(data)

    if Comm_rank(comm) == root
        # pass to the other threads the dst (reduced array)
        comm.passthrough[root] = convert(UInt, pointer(data))

        sync(comm.barrier)
    else
        # wait for thread 1 to compute the sum
        sync(comm.barrier)

        ptr = convert(Ptr{T}, comm.passthrough[root])
        arr = UnsafeArray(ptr, sz)
        data .= arr
    end
    return data
end
