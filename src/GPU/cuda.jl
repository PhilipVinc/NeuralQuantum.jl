using CuArrays: CuArrays, CuArray, @cufunc, CUBLAS, CUDAnative
using CuArrays: CuArrays.GPUArrays.GPUArray, @cufunc
using Base: ReshapedArray


@cufunc NeuralQuantum.ℒ(x) = one(x) + exp(x)
@cufunc NeuralQuantum.logℒ(x::Real) = log1p(exp(x)) #_gpu_logℒ(x)
@cufunc NeuralQuantum.logℒ(x::Complex) = log(one(x) + exp(x)) #_gpu_logℒ(x)
@cufunc NeuralQuantum.∂logℒ(x) = one(x)/(one(x)+exp(-x))

@cufunc NeuralQuantum.ℒ2(x)  = 2*cosh(x)
@cufunc NeuralQuantum.logℒ2(x)  = logℒ2(real(x)) + log(cos(imag(x)) +
    im * tanh(real(x)) * sin(imag(x)))
@cufunc NeuralQuantum.∂logℒ2(x) = tanh(x)

@cufunc fwd_der(f::typeof(NeuralQuantum.logℒ), x) = NeuralQuantum.∂logℒ(x)
@cufunc fwd_der(f::typeof(NeuralQuantum.logℒ2), x) = NeuralQuantum.∂logℒ2(x)

#_gpu_logℒ(x) = log1p(exp(x))
#@cufunc _gpu_logℒ(x::Real) = log1p(exp(x))
#@cufunc _gpu_logℒ(x::Complex) = log(one(x) + exp(x))


function build_rng_generator_T(arrT::CuArray, seed)
    return CuArrays.CURAND.generator()
end


@inline function NeuralQuantum._batched_outer_prod!(R::ReshapedArray,
    vb::CuArray, wb::CuArray)
    T=eltype(vb)
    dims_all = R.dims
    under    = R.parent
    under_indices = under.indices
    start = first(under_indices[1])
    n_batches = length(under_indices[2])
    batch_size = size(under.parent, 1)

    # Absolutely unsafe code, but avoids allocating a lot of views and pressuring
    # the GC (probably since v2.0 of CUDAnative it's useless)
    # does pointer aritmetic by hand
    Ri = view(R.parent.parent, :,:)
    Ri.dims = (size(R,1), size(R,2))
    vbi = view(vb, :, 1)
    wbi = view(wb, :, 1)

    for i=1:size(R, 3)
        fill!(Ri, 0)
        CUBLAS.ger!(one(T), vbi, wbi, Ri)

        Ri.ptr += Base.elsize(Ri)*batch_size
        vbi.ptr += Base.elsize(vbi)*length(vbi)
        wbi.ptr += Base.elsize(wbi)*length(wbi)
    end
    return R
end

@inline function NeuralQuantum._batched_outer_prod!(R::ReshapedArray, α,
    vb::CuArray, wb::CuArray)
    T=eltype(vb)
    dims_all = R.dims
    under    = R.parent
    under_indices = under.indices
    start = first(under_indices[1])
    n_batches = length(under_indices[2])
    batch_size = size(under.parent, 1)

    # Absolutely unsafe code, but avoids allocating a lot of views and pressuring
    # the GC (probably since v2.0 of CUDAnative it's useless)
    # does pointer aritmetic by hand
    Ri = view(R.parent.parent, :,:)
    Ri.dims = (size(R,1), size(R,2))
    vbi = view(vb, :, 1)
    wbi = view(wb, :, 1)

    for i=1:size(R, 3)
        fill!(Ri, 0)
        CUBLAS.ger!(T(α), vbi, wbi, Ri)

        Ri.ptr += Base.elsize(Ri)*batch_size
        vbi.ptr += Base.elsize(vbi)*length(vbi)
        wbi.ptr += Base.elsize(wbi)*length(wbi)
    end
    return R
end

@inline function NeuralQuantum._batched_outer_prod_∑!(R::ReshapedArray, α,
    vb::CuArray, wb::CuArray, vb2::CuArray, wb2::CuArray)
    T=eltype(vb)
    dims_all = R.dims
    under    = R.parent
    under_indices = under.indices
    start = first(under_indices[1])
    n_batches = length(under_indices[2])
    batch_size = size(under.parent, 1)

    # Absolutely unsafe code, but avoids allocating a lot of views and pressuring
    # the GC (probably since v2.0 of CUDAnative it's useless)
    # does pointer aritmetic by hand
    Ri = view(R.parent.parent, :,:)
    Ri.dims = (size(R,1), size(R,2))
    vbi = view(vb, :, 1)
    wbi = view(wb, :, 1)
    vb2i = view(vb2, :, 1)
    wb2i = view(wb2, :, 1)

    for i=1:size(R, 3)
        fill!(Ri, 0)
        CUBLAS.ger!(T(α), vbi, wbi, Ri)
        CUBLAS.ger!(T(α), vb2i, wb2i, Ri)

        Ri.ptr += Base.elsize(Ri)*batch_size
        vbi.ptr += Base.elsize(vbi)*length(vbi)
        wbi.ptr += Base.elsize(wbi)*length(wbi)
        vbi.ptr += Base.elsize(vb2i)*length(vb2i)
        wbi.ptr += Base.elsize(wb2i)*length(wb2i)
    end
    return R
end

@inline function NeuralQuantum._batched_outer_prod_Δ!(R::ReshapedArray, α,
    vb::CuArray, wb::CuArray, vb2::CuArray, wb2::CuArray)
    T=eltype(vb)
    dims_all = R.dims
    under    = R.parent
    under_indices = under.indices
    start = first(under_indices[1])
    n_batches = length(under_indices[2])
    batch_size = size(under.parent, 1)

    # Absolutely unsafe code, but avoids allocating a lot of views and pressuring
    # the GC (probably since v2.0 of CUDAnative it's useless)
    # does pointer aritmetic by hand
    Ri = view(R.parent.parent, :,:)
    Ri.dims = (size(R,1), size(R,2))
    vbi = view(vb, :, 1)
    wbi = view(wb, :, 1)
    vb2i = view(vb2, :, 1)
    wb2i = view(wb2, :, 1)

    for i=1:size(R, 3)
        fill!(Ri, 0)
        CUBLAS.ger!(T(α), vbi, wbi, Ri)
        CUBLAS.ger!(-T(α), vb2i, wb2i, Ri)

        Ri.ptr += Base.elsize(Ri)*batch_size
        vbi.ptr += Base.elsize(vbi)*length(vbi)
        wbi.ptr += Base.elsize(wbi)*length(wbi)
        vbi.ptr += Base.elsize(vb2i)*length(vb2i)
        wbi.ptr += Base.elsize(wb2i)*length(wb2i)
    end
    return R
end
