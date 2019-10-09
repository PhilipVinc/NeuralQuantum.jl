mutable struct LocalGradAccumulator{a,b,B,C,D,S,Ac} <: AbstractAccumulator
    cur_ψval::a     # The last value seen of <σ|ψ>
    cur_∇ψ::b
    ψ_counter::B    # Stores a counter, referring to how many non zero
                    # elements referring to σ we have found
    cur_ψ::Int      # Stores a counter, referring to which i-th element
                    # of the above array we are currently processing

    Oloc::C         # ⟨σ|Ô|ψ⟩ computed
    ∇Oloc::D        # ⟨σ|Ô|ψ⟩ computed
    n_tot::Int      # total number of σ' matrix element computed

    acc::Ac         # The accumulator
    σ::S            # just a temporary state to perform operation, cached
end

function LocalGradAccumulator(net, σ, n_tot, batch_sz)
    IT        = input_type(net)
    OT        = out_type(net)
    CT        = Complex{real(out_type(net))}
    f         = trainable_first(net)

    cur_ψval  = zero(OT)
    cur_∇ψ    = grad_cache(net)
    ψ_counter = zeros(Int, n_tot)
    Oloc      = similar(f, CT, n_tot)
    ∇Oloc     = grad_cache(CT, net, n_tot)

    _σ        = deepcopy(σ)
    accum     = GradientBatchAccumulator(net, σ, batch_sz)

    return LocalGradAccumulator(cur_ψval, cur_∇ψ, ψ_counter, 0,
                               Oloc, ∇Oloc, 0,
                               accum, _σ)
end

batch_size(a::LocalGradAccumulator) = length(a.acc)

function Base.resize!(c::LocalGradAccumulator, n_tot)
    resize!(c.Oloc, n_tot)

    CT = eltype(vec_data(c.∇Oloc)[1])
    c.∇Oloc  = grad_cache(CT, c.acc.bnet, n_tot)

    resize!(c.ψ_counter, n_tot)

    init!(c)
    return c
end

function init!(c::LocalGradAccumulator)
    c.cur_ψ  = 0

    c.Oloc  .= 0.0

    for v=vec_data(c.∇Oloc)
        v.= 0.0
    end

    c.ψ_counter .= 0

    init!(c.acc)
    return c
end

function Base.push!(c::LocalGradAccumulator, ψval::Number, ∇val)
    c.cur_ψ               += 1
    c.ψ_counter[c.cur_ψ]   = 0
    c.cur_ψval             = ψval
    #c.cur_∇ψ.tuple_all_weights[1] .= ∇val
    copyto!(c.cur_∇ψ.tuple_all_weights[1], ∇val)
    return c
end

function (c::LocalGradAccumulator)(mel, cngs, v)
    i = c.cur_ψ

    # If we have no changes, simply add the element to ⟨σ|Ô|ψ⟩ because
    # exp(logψ(σ)-logψ(σ)) = exp(0) = 1
    if length(cngs) == 0
        c.Oloc[i]  += mel
        #c.∇Oloc[i] += 0
        c.n_tot   +=  1
    else
        c.ψ_counter[i] += 1

        σ = set_index!(c.σ, index(v))
        apply!(σ, cngs)

        c.acc(mel, config(σ), c.cur_ψval, c.cur_∇ψ)
        isfull(c.acc) && process_buffer!(c)
    end

    return c
end

finalize!(c::LocalGradAccumulator) =
    process_buffer!(c, c.acc.buf_n)

function process_buffer!(c::LocalGradAccumulator, k=length(c.acc))
    out, ∇out = process_accumulator!(c.acc)
    #collect ? if using the gpu... need to think about this

    # Unsafe stuff can't be returned!
    ∇Oloc = uview(vec_data(c.∇Oloc)[1])
    ∇out  = uview(vec_data(∇out)[1])

    i = c.cur_ψ
    #@uviews ∇Oloc ∇out begin
        while k>0
            for j=1:c.ψ_counter[i]
                c.Oloc[i]  += out[k]
                #∇Oloc[:,i] .+= ∇out[:,k]
                view(∇Oloc, :, i) .+= view(∇out, :,k)
                k -= 1
                c.ψ_counter[i] = c.ψ_counter[i] - 1
                c.n_tot   +=  1
            end
            i -= 1
        end
    #end

    return c
end
