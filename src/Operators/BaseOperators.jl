abstract type AbsLinearOperator{T,N} #=<: AbstractArray{T,N} end=# end
const AbsLinearOp{T} = AbsLinearOperator{T,2}
const AbsLinearSuperOperator{T} = AbsLinearOperator{T,4}

abstract type AbsOpConnection end

Base.eltype(c::AbsOpConnection) = eltype(typeof(c))

# Random sampler for an abstract OpConnection (assuming it implements getindex and length)
function Random.rand(rng::AbstractRNG, sampl::Random.SamplerTrivial{T}) where {T<:NeuralQuantum.AbsOpConnection}
    conns = sampl[]
    N = length(conns)
    i = rand(rng, 1:N)
    @inbounds res = conns[i]
    return res
end


"""
    row_valdiff(op::AbsLinearOperator, v) -> OpConnection

Returns all non-zero elements in the row represented by the state `v` of
operator `op`. The result is an `OpConnection` type.
"""
row_valdiff(op::AbsLinearOperator, v) = row_valdiff!(conn_type(op)(), op, v, init=false)

"""
    row_valdiff!(opconn::OpConnection, op::AbsLinearOperator, v; init=true)

Returns all non-zero elements in the row represented by the state `v` of
operator `op` by mutating in-place the `OpConnection` object `opconn`
"""
function row_valdiff!(opconn, op, v; init=true)
    init && resize!(opconn, 0)
    _row_valdiff!(opconn, op, v)
end

function row_valstate(op::AbsLinearOperator, v)
    conns = row_valdiff(op, v)

    vp    = state_similar(v, length(conns))
    state_copy!(vp, v)

    T     = eltype(conns).parameters[1]
    mels  = zeros(T, length(conns))

    for (i,(mel, cngs)) = enumerate(conns)
        mels[i] = mel
        apply!(unsafe_get_batch(vp,i), cngs)
    end
    return mels, vp
end

row_valdiff_index(op::AbsLinearOperator, v) = row_valdiff_index!(OpConnectionIndex(op), op, v)
function row_valdiff_index! end

# standard functions
Base.:*(op::AbsLinearOperator, α::Number) = α*op
Base.:/(op::AbsLinearOperator, α::Number) = inv(α)*op

conn_type(op::AbsLinearOperator) = conn_type(typeof(op))
## AbstractArray Interface
