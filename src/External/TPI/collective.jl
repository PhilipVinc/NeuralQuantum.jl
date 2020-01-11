function Allreduce(src::Number, op, comm::Comm)
    src_arr = [src]
    Allreduce!(src_arr, +, comm)
    return src_arr[1]
end

Allreduce!(dst::A, op, comm::Comm) where {T, A<:DenseArray{T}} =
    Allreduce!(dst, dst, op, comm)

function Allreduce!(dst::A, src::A, op, comm::Comm) where {T, A<:DenseArray{T}}
    sz = size(src)

    if Threads.threadid() == 1
        # pass to the other threads the dst (reduced array)
        comm.passthrough[Threads.threadid()] = convert(UInt, pointer(dst))

        dst !== src && copyto!(dst, src)

        # wait for everyone to have written their data
        sync(comm.barrier)

        # sum everything in place
        for (id, addr) in enumerate(comm.passthrough)
            id == Threads.threadid() && continue

            ptr = convert(Ptr{T}, addr)
            arr = UnsafeArray(ptr, sz)
            broadcast!(op, dst, dst, arr)
        end

        sync(comm.barrier)
    else
        # pass to the master thread the src array
        comm.passthrough[Threads.threadid()] = convert(UInt, pointer(src))

        # report to thread 1 that you set the pointer
        sync(comm.barrier)

        # wait for thread 1 to compute the sum
        sync(comm.barrier)

        ptr = convert(Ptr{T}, comm.passthrough[1])
        arr = UnsafeArray(ptr, sz)
        copyto!(dst, arr)

    end
    
    sync(comm.barrier)
    return dst
end

Allgatherv!(dst::A, counts, comm::Comm) where {T, A<:DenseArray{T}} =
     Allgatherv!(dst, dst, counts, comm)

function Allgatherv!(dst::A, src::B, counts, comm::Comm) where {T, A<:DenseArray{T}, B<:DenseArray{T}}
    sz = length(dst)
    id = Threads.threadid()

    sum(counts) <= length(dst) || throw(error("Destination buffer too small"))

    # Share among all threads the destination pointer
    comm.passthrough[id] = convert(UInt, pointer(dst))
    sync(comm.barrier)

    # Compute the offset to write to in the destination
    Δ = 1
    for i=1:id-1
        Δ += counts[i]
    end

    Δ_src = 1
    if src === dst
        Δ_src = Δ
    end

    # Write
    for (i, addr) in enumerate(comm.passthrough)
        dst === src && i == id && continue

        ptr = convert(Ptr{T}, addr)
        dst = UnsafeArray(ptr, (length(dst), ))

        copyto!(dst, Δ, src, Δ_src, counts[id])
    end

    sync(comm.barrier)

    return dst
end
