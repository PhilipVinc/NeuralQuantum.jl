mutable struct LocalComputator{A,B,C,D,E,F,N}
    ψ_vals::A       # Stores all the values <σ|ψ> computed for this chain
    ψ_counter::B    # Stores a counter, referring to how many non zero
                    # elements referring to σ we have found
    cur_ψ::Int      # Stores a counter, referring to which i-th element
                    # of the above array we are currently processing

    Oloc::C         # ⟨σ|Ô|ψ⟩ computed
    n_tot::Int      # total number of σ' matrix element computed

    mel_buf::D      # ⟨σ|Ô|σ'⟩ in the buffer
    ψ0_buf::A       # Buffers alls the <σ|ψ> of the denominator
    v_buf::F        # the matrix of Nsites x batchsz used as state
    buf_n::Int      # Counter for elements in buffer

    σ::E            # just a temporary state to perform operation, cached
    batch_sz::Int
    bnet::N
end

function LocalComputator(net, σ, batch_sz)
    IT        = input_type(net)
    OT        = out_type(net)
    f         = trainable_first(net)

    ψ_vals    = similar(f, OT, 1, 2)
    ψ0_buf    = similar(f, OT, 1, batch_sz)
    ψ_counter = zeros(Int, batch_sz)
    Oloc      = similar(f, OT, batch_sz)

    mel_buf   = zeros(OT, batch_sz)
    _σ        = deepcopy(σ)
    v         = similar(f, IT, length(config(σ)), batch_sz)

    return LocalComputator(ψ_vals, ψ_counter, 0,
                           Oloc, 0,
                           mel_buf, ψ0_buf, v, 0,
                           _σ, batch_sz, cached(net, batch_sz))
end

function init!(c::LocalComputator, ψ_vals)
    c.ψ_vals = ψ_vals
    c.cur_ψ  = 0
    c.buf_n  = 0
    c.Oloc   = similar(ψ_vals, length(ψ_vals))
    c.Oloc  .= 0.0
    c.ψ_counter = zeros(Int, length(ψ_vals))
    return c
end

function Base.push!(c::LocalComputator, ψval::Number)
    c.cur_ψ             += 1
    c.ψ_counter[c.cur_ψ] = 0
    return c
end

function (c::LocalComputator)(mel, cngs, v)
    i = c.cur_ψ

    # If we have no changes, simply add the element to ⟨σ|Ô|ψ⟩ because
    # exp(logψ(σ)-logψ(σ)) = exp(0) = 1
    if length(cngs) == 0
        c.Oloc[i] += mel
        c.n_tot   +=  1
    else
        c.buf_n = c.buf_n + 1

        c.ψ_counter[i] += 1
        c.ψ0_buf[c.buf_n] = c.ψ_vals[i]
        apply!(v, cngs)
        c.mel_buf[c.buf_n] = mel
        c.v_buf[:,c.buf_n] .= config(v)
        c.buf_n == c.batch_sz && process_buffer!(c)
    end

    return c
end

finalize!(c::LocalComputator) =
    process_buffer!(c, c.buf_n)

function process_buffer!(c::LocalComputator, k=c.batch_sz)
    net = c.bnet

    out = net(c.v_buf)
    out .-= c.ψ0_buf
    out .= exp.(out)
    #collect

    i = c.cur_ψ
    while k>0
        for j=1:c.ψ_counter[i]
            c.Oloc[i] += out[k] * c.mel_buf[k]
            k -= 1
            c.ψ_counter[i] = c.ψ_counter[i] - 1
            c.n_tot   +=  1
        end
        i -= 1
    end

    c.buf_n = 0
    return c
end
