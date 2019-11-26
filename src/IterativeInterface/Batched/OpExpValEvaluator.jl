mutable struct OpExpEvaluator{a,B,C,S,Ac} <: AbstractAccumulator
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

function OpExpEvaluator(net, σ, n_tot, batch_sz)
    IT        = input_type(net)
    OT        = out_type(net)
    f         = trainable_first(net)

    cur_ψval  = zero(OT)
    ψ_counter = zeros(Int, n_tot)
    Oloc      = similar(f, OT, n_tot)

    _σ        = deepcopy(σ)
    accum     = ScalarBatchAccumulator(net, σ, batch_sz)

    return OpExpEvaluator(cur_ψval, ψ_counter, 0,
                               Oloc, 0,
                               accum, _σ)
end

batch_size(a::OpExpEvaluator) = length(a.acc)

function Base.resize!(c::OpExpEvaluator, n_tot)
    resize!(c.Oloc, n_tot)

    resize!(c.ψ_counter, n_tot)

    init!(c)
    return c
end

function init!(c::OpExpEvaluator)
    c.cur_ψ  = 0
    c.n_tot  = 0

    c.Oloc  .= 0.0
    c.ψ_counter  .= 0

    init!(c.acc)
    return c
end

@inline Base.push!(c::OpExpEvaluator, ψval::Number, 𝛁val) =
    push!(c, ψval)

function Base.push!(c::OpExpEvaluator, ψval::Number)
    c.cur_ψ             += 1
    c.ψ_counter[c.cur_ψ] = 0
    c.cur_ψval           = ψval
    return c
end

# I don't really need two versions of this command,
# but Julia is stupid so I need.
function (c::OpExpEvaluator)(mel::Number, cngs_l,
                             cngs_r, v::State)
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

function (c::OpExpEvaluator)(mel::Number, cngs::StateChanges, v::State)
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

function _send_to_accumulator(c::OpExpEvaluator, mel, σ)
    c.ψ_counter[c.cur_ψ] += 1

    cσ = config(σ)
    c.acc(mel, config(σ), c.cur_ψval)
    isfull(c.acc) && process_buffer!(c)
    return c
end


finalize!(c::OpExpEvaluator) =
    process_buffer!(c, c.acc.buf_n)

function process_buffer!(c::OpExpEvaluator, k=length(c.acc))
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
