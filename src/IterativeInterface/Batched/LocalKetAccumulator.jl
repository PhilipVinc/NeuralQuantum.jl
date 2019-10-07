mutable struct LocalKetAccumulator{a,B,C,S,Ac} <: AbstractAccumulator
    cur_ψval::a     # The last value seen of <σ|ψ>
    ψ_counter::B    # Stores a counter, referring to how many non zero
                    # elements referring to σ we have found
    cur_ψ::Int      # Stores a counter, referring to which i-th element
                    # of the above array we are currently processing

    Oloc::C         # ⟨σ|Ô|ψ⟩ computed
    n_tot::Int      # total number of σ' matrix element computed

    acc::Ac         # The accumulator
    σ::S            # just a temporary state to perform operation, cached
end

function LocalKetAccumulator(net, σ, batch_sz)
    IT        = input_type(net)
    OT        = out_type(net)
    f         = trainable_first(net)

    cur_ψval  = zero(OT)
    ψ_counter = zeros(Int, 1)
    Oloc      = similar(f, OT, batch_sz)

    _σ        = deepcopy(σ)
    accum     = ScalarBatchAccumulator(net, σ, batch_sz)

    return LocalKetAccumulator(cur_ψval, ψ_counter, 0,
                               Oloc, 0,
                               accum, _σ)
end

batch_size(a::LocalKetAccumulator) = length(a.acc)

function init!(c::LocalKetAccumulator, ψ_vals)
    c.cur_ψ  = 0
    c.Oloc   = similar(ψ_vals, length(ψ_vals))
    c.Oloc  .= 0.0
    c.ψ_counter = zeros(Int, length(ψ_vals))
    init!(c.acc)
    return c
end

function Base.push!(c::LocalKetAccumulator, ψval::Number)
    c.cur_ψ             += 1
    c.ψ_counter[c.cur_ψ] = 0
    c.cur_ψval           = ψval
    return c
end

function (c::LocalKetAccumulator)(mel, cngs, v)
    i = c.cur_ψ

    # If we have no changes, simply add the element to ⟨σ|Ô|ψ⟩ because
    # exp(logψ(σ)-logψ(σ)) = exp(0) = 1
    if length(cngs) == 0
        c.Oloc[i] += mel
        c.n_tot   +=  1
    else
        c.ψ_counter[i] += 1

        σ = set_index!(c.σ, index(v))
        apply!(σ, cngs)

        c.acc(mel, config(σ), c.cur_ψval)
        isfull(c.acc) && process_buffer!(c)
    end

    return c
end

finalize!(c::LocalKetAccumulator) =
    process_buffer!(c, c.acc.buf_n)

function process_buffer!(c::LocalKetAccumulator, k=length(c.acc))
    out = process_accumulator!(c.acc)
    #collect ? if using the gpu... need to think about this

    i = c.cur_ψ
    while k>0
        for j=1:c.ψ_counter[i]
            c.Oloc[i] += out[k]
            k -= 1
            c.ψ_counter[i] = c.ψ_counter[i] - 1
            c.n_tot   +=  1
        end
        i -= 1
    end

    return c
end
