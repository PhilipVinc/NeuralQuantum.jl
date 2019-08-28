struct OpConnectionIndex{A<:AbstractArray,B<:AbstractArray}
    mel::A
    indices::B
end

function OpConnectionIndex(op::AbsLinearOperator)
    t = conn_type(op)
    v1, v2, v3 = t.parameters
    return OpConnectionIndex(v1(), Vector{Int}())
end

function Base.resize!(c::OpConnectionIndex, n)
    resize!(c.mel,        n)
    resize!(c.indices,  n)
end

Base.eltype(c::OpConnectionIndex{A,B}) where {A,B} = (eltype(A), eltype(B))
Base.length(c::OpConnectionIndex) = length(c.mel)
Base.size(c::OpConnectionIndex) = size(c.mel)

function Base.push!(c::OpConnectionIndex, (m_els, indices))
    push!(c.mel, m_els)
    push!(c.indices, indices)
    return c
end

function Base.append!(c::OpConnectionIndex, (m_els, indices))
    append!(c.mel, m_els)
    append!(c.indices, indices)
    return c
end

Base.getindex(c::OpConnectionIndex, i) = (c.mel[i], c.indices[i])

function Base.iterate(iter::OpConnectionIndex, state=1)
    state > length(iter) && return nothing
    return (iter[state], state + 1)
end
