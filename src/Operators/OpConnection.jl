struct OpConnection{A<:AbstractArray,B<:AbstractArray,C<:AbstractArray}
    mel::A
    to_change::B
    new_values::C
end

function OpConnection(op::AbsLinearOperator)
    t = conn_type(op)
    v1, v2, v3 = t.parameters
    return OpConnection(v1(), v2(), v3())
end

function Base.resize!(c::OpConnection, n)
    resize!(c.mel,        n)
    resize!(c.to_change,  n)
    resize!(c.new_values, n)
end

Base.eltype(c::OpConnection{A,B,C}) where {A,B,C} = (eltype(A), eltype(B), eltype(C))
Base.length(c::OpConnection) = length(c.mel)
Base.size(c::OpConnection) = size(c.mel)

function Base.push!(c::OpConnection, (m_els, to_change, new_values))
    push!(c.mel, m_els)
    push!(c.to_change, to_change)
    push!(c.new_values, new_values)
    c
end

function Base.append!(c::OpConnection, (m_els, to_change, new_values))
    println(m_els)

    append!(c.mel, m_els)
    append!(c.to_change, to_change)
    append!(c.new_values, new_values)
    c
end

Base.getindex(c::OpConnection, i) = (c.mel[i], c.to_change[i], c.new_values[i])

function Base.iterate(iter::OpConnection, state=1)
    state > length(iter) && return nothing
    return (iter[state], state + 1)
end
