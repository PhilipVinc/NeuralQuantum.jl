mutable struct GradientBatchAccumulator{N,A,B,C,D,E,F}
    bnet::N

    in_buf::A       # the matrix of Nsites x batchsz used as state
    out_buf::B
    out2_buf::B
    ∇out_buf::F

    ψ0_buf::C       # Buffers alls the <σ|ψ> of the denominator
    ∇0_buf::E
    mel_buf::D      # ⟨σ|Ô|σ'⟩ in the buffer

    buf_n::Int      # Counter for elements in buffer
    batch_sz::Int
end

function GradientBatchAccumulator(net::NeuralNetwork, v::State, batch_sz)
    bnet     = cached(net, batch_sz)
    CT       = Complex{real(out_type(net))}

    w        = trainable_first(net)
    RT       = real(eltype(w))
    in_buf   = preallocate_state_batch(w, RT, v, batch_sz)
    out_buf  = similar(w, CT, 1, batch_sz)
    out2_buf = similar(w, CT, 1, batch_sz)
    ∇out_buf = grad_cache(CT, net, batch_sz)

    ψ0_buf = similar(w, out_type(net), 1, batch_sz)
    ∇0_buf = grad_cache(net, batch_sz)
    mel_buf = similar(w, CT, 1, batch_sz)

    return GradientBatchAccumulator(
        bnet, in_buf, out_buf, out2_buf, ∇out_buf,
        ψ0_buf, ∇0_buf, mel_buf, 0, batch_sz)
end

Base.length(a::GradientBatchAccumulator) = a.batch_sz

isfull(a::GradientBatchAccumulator) = a.buf_n == length(a)

init!(c::GradientBatchAccumulator) = c.buf_n = 0

function (c::GradientBatchAccumulator)(mel, v, ψ0, ∇0_buf)
    # Increase the step in our internal buffer
    # this is guaranteed to always be < max_capacity
    c.buf_n = c.buf_n + 1

    c.ψ0_buf[c.buf_n]   = ψ0
    c.mel_buf[c.buf_n]  = mel

    store_state!(c.in_buf, v, c.buf_n)
    c∇0_buf = vec_data(c.∇0_buf)[1]
    dd = vec_data(∇0_buf)[1]
    @uviews c∇0_buf dd begin
        c∇0_buf[:,c.buf_n] .= dd
    end
end

function process_accumulator!(c::GradientBatchAccumulator)
    out_buf   = c.out_buf
    ∇out      = c.∇out_buf
    init!(c)

    logψ_and_∇logψ!(c.∇out_buf, c.out_buf, c.bnet, c.in_buf)
    #out_buf .-= c.ψ0_buf      #logΔ
    #out_buf  .= exp.(out_buf) #exp(logΔ)
    #out_buf .*= c.mel_buf

    c.out2_buf .= c.mel_buf .* exp.(out_buf .- c.ψ0_buf)
    #collect ? if using the gpu... need to think about this

    ∇out = vec_data(c.∇out_buf)[1]
    ∇0   = vec_data(c.∇0_buf)[1]

    ∇out .-= ∇0
    ∇out .*= c.mel_buf

    return c.out2_buf, c.∇out_buf
end
