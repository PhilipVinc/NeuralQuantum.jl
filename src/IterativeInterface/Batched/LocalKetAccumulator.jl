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

function init!(c::LocalKetAccumulator, chain_len::Int)
    c.cur_ψ  = 0
    resize!(c.Oloc, chain_len)
    c.Oloc  .= 0.0
    resize!(c.ψ_counter, chain_len)
    c.ψ_counter  .= 0

    init!(c.acc)
    return c
end

function Base.push!(c::LocalKetAccumulator, ψval::Number)
    c.cur_ψ             += 1
    c.ψ_counter[c.cur_ψ] = 0
    c.cur_ψval           = ψval
    return c
end

# I don't really need two versions of this command,
# but Julia is stupid so I need.
function (c::LocalKetAccumulator)(mel::Number, cngs_l, cngs_r, v::State)
    n_cngs_l = isnothing(cngs_l) ? 0 : length(cngs_l)
    n_cngs_r = isnothing(cngs_r) ? 0 : length(cngs_r)

    mel == 0.0 && return c

    # If we have no changes, simply add the element to ⟨σ|Ô|ψ⟩ because
    # exp(logψ(σ)-logψ(σ)) = exp(0) = 1
    if n_cngs_l == 0 && n_cngs_r == 0
        c.Oloc[c.cur_ψ] += mel
        c.n_tot   +=  1
    else
        σ = set_index!(c.σ, index(v))
        apply!(σ, cngs_l, cngs_r)
        _send_to_accumulator(c, mel, σ)
    end
    return c
end

function (c::LocalKetAccumulator)(mel::Number, cngs::StateChanges, v::State)
    mel == 0.0 && return c

    # If we have no changes, simply add the element to ⟨σ|Ô|ψ⟩ because
    # exp(logψ(σ)-logψ(σ)) = exp(0) = 1
    if length(cngs) == 0
        c.Oloc[c.cur_ψ] += mel
        c.n_tot   +=  1
    else
        σ = set_index!(c.σ, index(v))
        apply!(σ, cngs)
        _send_to_accumulator(c, mel, σ)
    end
    return c
end

function _send_to_accumulator(c::LocalKetAccumulator, mel, σ)
    c.ψ_counter[c.cur_ψ] += 1

    cσ = config(σ)
    c.acc(mel, config(σ), c.cur_ψval)
    isfull(c.acc) && process_buffer!(c)
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
