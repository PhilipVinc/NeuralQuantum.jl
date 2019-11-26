using CuArrays: CuArrays, CuArray, @cufunc, CUBLAS
using CuArrays: CuArrays.GPUArrays.GPUArray
using Base: ReshapedArray


@cufunc NeuralQuantum.ℒ(x) = one(x) + exp(x)
@cufunc NeuralQuantum.∂logℒ(x) = one(x)/(one(x)+exp(-x))

#_gpu_logℒ(x) = log1p(exp(x))
#@cufunc _gpu_logℒ(x::Real) = log1p(exp(x))
#@cufunc _gpu_logℒ(x::Complex) = log(one(x) + exp(x))

@cufunc NeuralQuantum.logℒ(x) = log(one(x) + exp(x)) #_gpu_logℒ(x)


@inline function NeuralQuantum._batched_outer_prod!(R::ReshapedArray,
    vb::CuArray, wb::CuArray)
    T=eltype(vb)
    dims_all = R.dims
    under    = R.parent
    under_indices = under.indices
    start = first(under_indices[1])
    n_batches = length(under_indices[2])
    batch_size = size(under.parent, 1)

    Ri  = CuArray{eltype(R),2}(under.parent.buf, dims_all[1:2], own=false)
    vbi = CuArray{eltype(vb),1}(vb.buf, (size(vb, 1),), own=false)
    wbi = CuArray{eltype(wb),1}(wb.buf, (size(wb, 1),), own=false)

    for i=1:size(R, 3)
        Ri.offset = (i-1)*Base.elsize(Ri)*batch_size
        vbi.offset = (i-1)*Base.elsize(vbi)*length(vbi)
        wbi.offset = (i-1)*Base.elsize(wbi)*length(wbi)
        fill!(Ri, 0)
        CUBLAS.ger!(one(T), vbi, wbi, Ri)
    end
    return R
end

@inline function NeuralQuantum._batched_outer_prod!(R::ReshapedArray, α,
    vb::GPUArray, wb::GPUArray)
    T=eltype(vb)
    dims_all = R.dims
    under    = R.parent
    under_indices = under.indices
    start = first(under_indices[1])
    n_batches = length(under_indices[2])
    batch_size = size(under.parent, 1)

    Ri  = CuArray{eltype(R),2}( under.parent.buf, dims_all[1:2], own=false)
    vbi = CuArray{eltype(vb),1}(vb.buf, (size(vb, 1),), own=false)
    wbi = CuArray{eltype(wb),1}(wb.buf, (size(wb, 1),), own=false)

    for i=1:size(R, 3)
        Ri.offset = (i-1)*Base.elsize(Ri)*batch_size
        vbi.offset = (i-1)*Base.elsize(vbi)*length(vbi)
        wbi.offset = (i-1)*Base.elsize(wbi)*length(wbi)
        fill!(Ri, 0)
        CUBLAS.ger!(T(α), vbi, wbi, Ri)
    end
    return R
end

@inline function NeuralQuantum._batched_outer_prod_∑!(R::ReshapedArray, α,
    vb::GPUArray, wb::GPUArray, vb2::GPUArray, wb2::GPUArray)
    T=eltype(vb)
    dims_all = R.dims
    under    = R.parent
    under_indices = under.indices
    start = first(under_indices[1])
    n_batches = length(under_indices[2])
    batch_size = size(under.parent, 1)

    Ri   = CuArray{eltype(R),2}( under.parent.buf, dims_all[1:2], own=false)
    vbi  = CuArray{eltype(vb),1}(vb.buf, (size(vb, 1),), own=false)
    wbi  = CuArray{eltype(wb),1}(wb.buf, (size(wb, 1),), own=false)

    vb2i = CuArray{eltype(R),1}(vb2.buf, (size(vb2, 1),), own=false)
    wb2i = CuArray{eltype(R),1}(wb2.buf, (size(wb2, 1),), own=false)

    for i=1:size(R, 3)
        Ri.offset = (i-1)*Base.elsize(Ri)*batch_size
        vbi.offset = (i-1)*Base.elsize(vbi)*length(vbi)
        wbi.offset = (i-1)*Base.elsize(wbi)*length(wbi)
        vb2i.offset = (i-1)*Base.elsize(vb2i)*length(vb2i)
        wb2i.offset = (i-1)*Base.elsize(wb2i)*length(wb2i)
        fill!(Ri, 0)
        CUBLAS.ger!(T(α), vbi, wbi, Ri)
        CUBLAS.ger!(T(α), vb2i, wb2i, Ri)
    end
    return R
end

@inline function NeuralQuantum._batched_outer_prod_Δ!(R::ReshapedArray, α,
    vb::GPUArray, wb::GPUArray, vb2::GPUArray, wb2::GPUArray)
    T=eltype(vb)
    dims_all = R.dims
    under    = R.parent
    under_indices = under.indices
    start = first(under_indices[1])
    n_batches = length(under_indices[2])
    batch_size = size(under.parent, 1)

    Ri   = CuArray{eltype(R),2}( under.parent.buf, dims_all[1:2], own=false)
    vbi  = CuArray{eltype(vb),1}(vb.buf, (size(vb, 1),), own=false)
    wbi  = CuArray{eltype(wb),1}(wb.buf, (size(wb, 1),), own=false)

    vb2i = CuArray{eltype(vb2),1}(vb2.buf, (size(vb2, 1),), own=false)
    wb2i = CuArray{eltype(wb2),1}(wb2.buf, (size(wb2, 1),), own=false)

    for i=1:size(R, 3)
        Ri.offset = (i-1)*Base.elsize(Ri)*batch_size
        vbi.offset = (i-1)*Base.elsize(vbi)*length(vbi)
        wbi.offset = (i-1)*Base.elsize(wbi)*length(wbi)
        vb2i.offset = (i-1)*Base.elsize(vb2i)*length(vb2i)
        wb2i.offset = (i-1)*Base.elsize(wb2i)*length(wb2i)
        fill!(Ri, 0)
        CUBLAS.ger!(T(α), vbi, wbi, Ri)
        CUBLAS.ger!(-T(α), vb2i, wb2i, Ri)
    end
    return R
end
