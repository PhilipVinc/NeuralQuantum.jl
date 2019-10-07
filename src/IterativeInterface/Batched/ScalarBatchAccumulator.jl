mutable struct ScalarBatchAccumulator{N,A,B,C,D}
    bnet::N

    in_buf::A       # the matrix of Nsites x batchsz used as state
    out_buf::B

    ψ0_buf::C       # Buffers alls the <σ|ψ> of the denominator
    mel_buf::D      # ⟨σ|Ô|σ'⟩ in the buffer

    buf_n::Int      # Counter for elements in buffer
    batch_sz::Int
end

function ScalarBatchAccumulator(net::NeuralNetwork, v::State, batch_sz)
    bnet    = cached(net, batch_sz)

    w       = trainable_first(net)
    RT      = real(eltype(w))
    in_buf  = preallocate_state_batch(w, RT, v, batch_sz)
    out_buf = similar(w, out_type(net), 1, batch_sz)

    ψ0_buf  = similar(w, out_type(net), 1, batch_sz)
    mel_buf = similar(w, out_type(net), 1, batch_sz)

    return ScalarBatchAccumulator(
        bnet, in_buf, out_buf,
        ψ0_buf, mel_buf, 0, batch_sz)
end


Base.length(a::ScalarBatchAccumulator) = a.batch_sz

isfull(a::ScalarBatchAccumulator) = a.buf_n == length(a)

init!(c::ScalarBatchAccumulator) = c.buf_n = 0

function (c::ScalarBatchAccumulator)(mel, v, ψ0)
    # Increase the step in our internal buffer
    # this is guaranteed to always be < max_capacity
    c.buf_n = c.buf_n + 1

    c.ψ0_buf[c.buf_n]   = ψ0
    c.mel_buf[c.buf_n]  = mel
    store_state!(c.in_buf, v, c.buf_n)
end

function process_accumulator!(c::ScalarBatchAccumulator)
    out_buf  = logψ!(c.out_buf, c.bnet, c.in_buf)
    out_buf .-= c.ψ0_buf
    out_buf  .= exp.(out_buf)
    #collect ? if using the gpu... need to think about this

    out_buf .*= c.mel_buf

    c.buf_n = 0

    return out_buf
end
