using CuArrays: cudims, @cuda, cudaconvert, cufunction, mapreducedim_kernel_parallel
using CuArrays: CUDAnative, CUDAdrv, attribute, @cufunc

function Base._mapreducedim!(f, op, R::CuArray{T}, A::CuArray{T}) where {T}
    # the kernel as generated from `f` and `op` can require lots of registers (eg. #160),
    # so we need to be careful about how many threads we launch not to run out of them.
    Rlength = length(R)
    Ssize = ifelse.(size(R) .== 1, size(A), 1)
    Slength = prod(Ssize)
    CIS = CartesianIndices(Ssize)

    parallel_args = (f, op, R, A, CIS, Rlength, Slength)
    GC.@preserve parallel_args begin
        parallel_kargs = cudaconvert.(parallel_args)
        parallel_tt = Tuple{Core.Typeof.(parallel_kargs)...}
        parallel_kernel = cufunction(mapreducedim_kernel_parallel, parallel_tt)

        # we are limited in how many threads we can launch...
        ## by the kernel
        kernel_threads = CUDAnative.maxthreads(parallel_kernel)
        ## by the device
        dev = CUDAdrv.device()
        block_threads = (x=attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X),
                         y=attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y),
                         total=attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))

        # figure out a legal launch configuration
        y_thr = min(nextpow(2, Rlength ÷ 512 + 1), 512, block_threads.y, kernel_threads)
        x_thr = min(512 ÷ y_thr, Slength, block_threads.x,
                    ceil(Int, block_threads.total/y_thr),
                    ceil(Int, kernel_threads/y_thr))

        #if x_thr >= 8
            blk, thr = (Rlength - 1) ÷ y_thr + 1, (x_thr, y_thr, 1)
            parallel_kernel(parallel_kargs...; threads=thr, blocks=blk)
        #else
        #    # not enough work, fall back to serial reduction
        #    range = ifelse.(length.(axes(R)) .== 1, axes(A), nothing)
        #    blk, thr = cudims(R)
        #    @cuda(blocks=blk, threads=thr, mapreducedim_kernel_serial(f, op, R, A, range))
        #end
    end

    return R
end

# Fix my bug
@inline CUDAnative.exp(x::Complex{Float32}) = CUDAnative.exp(x.re) * (CUDAnative.cos(x.im) + 1.0f0im * CUDAnative.sin(x.im))
@inline CUDAnative.exp_fast(x::Complex{Float32}) = CUDAnative.exp_fast(x.re) * (CUDAnative.cos_fast(x.im) + 1.0f0im * CUDAnative.sin_fast(x.im))

#
function Statistics.mean!(y::CuVector, x::CuArray{T,3}) where T
    ỹ = reshape(y, length(y), 1, 1)
    mean!(ỹ, x)
    return y
end
