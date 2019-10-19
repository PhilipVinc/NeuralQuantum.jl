mutable struct GradientBatchAccumulator{N,A,B,C,C2,D,D2,E,F,B2,F2}
    bnet::N

    in_buf::A       # the matrix of Nsites x batchsz used as state
    out_buf::B
    ∇out_buf::F

    res::B2
    ∇res::F2

    ψ0_buf_g::C       # Buffers alls the <σ|ψ> of the denominator
    ψ0_buf_c::C2       # Buffers alls the <σ|ψ> of the denominator
    ∇0_buf::E
    mel_buf_c::D      # ⟨σ|Ô|σ'⟩ in the buffer
    mel_buf_g::D2      # ⟨σ|Ô|σ'⟩ in the buffer

    buf_n::Int      # Counter for elements in buffer
    batch_sz::Int
end

function GradientBatchAccumulator(net::NeuralNetwork, v::State, batch_sz)
    bnet     = cached(net, batch_sz)
    CT       = Complex{real(out_type(net))}

    w        = trainable_first(net)
    RT       = real(eltype(w))
    in_buf   = preallocate_state_batch(w, RT, v, batch_sz)
    out_buf  = similar(w, out_type(bnet), 1, batch_sz)
    ∇out_buf = grad_cache(net, batch_sz)

    res      = similar(w, CT, 1, batch_sz)
    ∇res     = grad_cache(CT, net, batch_sz)

    ψ0_buf_g  = similar(w, out_type(net), 1, batch_sz)
    ψ0_buf_c  = zeros(out_type(net), 1, batch_sz)
    ∇0_buf    = grad_cache(net, batch_sz)
    mel_buf_g = similar(w, CT, 1, batch_sz)
    mel_buf_c = zeros(CT, 1, batch_sz)

    if typeof(mel_buf_g) == typeof(mel_buf_c)
        mel_buf_g = nothing
        ψ0_buf_g = nothing
    end

    return GradientBatchAccumulator(
        bnet,
        in_buf, out_buf, ∇out_buf,
        res, ∇res,
        ψ0_buf_g, ψ0_buf_c,
        ∇0_buf,
        mel_buf_c, mel_buf_g,
        0, batch_sz)
end

Base.length(a::GradientBatchAccumulator) = a.batch_sz

isfull(a::GradientBatchAccumulator) = a.buf_n == length(a)

init!(c::GradientBatchAccumulator) = c.buf_n = 0

function (c::GradientBatchAccumulator)(mel, v, ψ0, ∇0_buf)
    # Increase the step in our internal buffer
    # this is guaranteed to always be < max_capacity
    c.buf_n = c.buf_n + 1

    c.ψ0_buf_c[ c.buf_n]   = ψ0
    c.mel_buf_c[c.buf_n]   = mel

    store_state!(c.in_buf, v, c.buf_n)
    c∇0_buf = vec_data(c.∇0_buf)[1]
    dd = vec_data(∇0_buf)[1]
    @uviews c∇0_buf dd begin
        c∇0_buf[:,c.buf_n] .= dd
    end
    return c
end

function process_accumulator!(c::GradientBatchAccumulator)
    out_buf   = c.out_buf
    ∇out      = c.∇out_buf
    init!(c)

    logψ_and_∇logψ!(∇out, out_buf, c.bnet, c.in_buf)
    #out_buf .-= c.ψ0_buf      #logΔ
    #out_buf  .= exp.(out_buf) #exp(logΔ)
    #out_buf .*= c.mel_buf
    ψ0_buf  = isnothing(c.ψ0_buf_g)  ? c.ψ0_buf_c : copy!(c.ψ0_buf_g, c.ψ0_buf_c)
    mel_buf = isnothing(c.mel_buf_g) ? c.ψ0_buf_c : copy!(c.mel_buf_g, c.mel_buf_c)

    c.res .= mel_buf .* exp.(out_buf .- ψ0_buf)
    #collect ? if using the gpu... need to think about this

    ∇res = vec_data(c.∇res)[1]
    ∇out = vec_data(∇out)[1]
    ∇0   = vec_data(c.∇0_buf)[1]

    ∇res .= mel_buf .* (∇out .- ∇0)

    return nothing
    #return c.out2_buf, c.∇out2_buf
end
