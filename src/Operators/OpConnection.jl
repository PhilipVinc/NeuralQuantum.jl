struct OpConnection{A<:AbstractArray,B<:AbstractArray,C<:AbstractArray}
    mel::A
    changes::Vector{StateChanges{B,C}}
end

function OpConnection(op::AbsLinearOperator)
    t = conn_type(op)
    v1, v2, v3 = t.parameters
    return OpConnection(v1(), Vector{StateChanges{v2,v3}}())
end

function OpConnection{A,B,C}() where {A,B,C}
    return OpConnection(A(), Vector{typeof(StateChanges(B(),C()))}())
end

function Base.resize!(c::OpConnection, n)
    resize!(c.mel,     n)
    resize!(c.changes, n)
    return c
end

Base.eltype(c::OpConnection{A,B,C}) where {A,B,C} = (eltype(A), eltype(B), eltype(C))
Base.length(c::OpConnection) = length(c.mel)
Base.size(c::OpConnection) = size(c.mel)

Base.:(==)(a::OpConnection, b::OpConnection) = (a.to_change == b.to_change &&
                                              a.new_values == b.new_values)

function Base.push!(c::OpConnection, (m_els, to_change, new_values))
    push!(c.mel, m_els)
    push!(c.changes, StateChanges(to_change, new_values))
    return c
end

function Base.push!(c::OpConnection, (m_els, cngs)::Tuple{Number, StateChanges})
    push!(c.mel, m_els)
    push!(c.changes, cngs)
    return c
end

function Base.append!(c::OpConnection, (m_els, to_change, new_values))
    append!(c.mel, m_els)
    cngs = Vector{StateChanges{eltype(to_change), eltype(new_values)}}()
    for (tcng, nwvls) = zip(to_change, new_values)
        push!(cngs, StateChanges(tcng, nwvls))
    end
    append!(c.changes, cngs)
    return c
end

function Base.append!(c::OpConnection, (m_els, cngs)::Tuple{Vector{<:Number},Vector{StateChanges}})
    append!(c.mel, m_els)
    append!(c.changes, cngs)
    return c
end

function Base.append!(c::OpConnection, new::OpConnection)
    append!(c.mel, new.mel)
    append!(c.changes, new.changes)
    return c
end

Base.getindex(c::OpConnection, i) = (c.mel[i], c.changes[i])

function Base.iterate(iter::OpConnection, state=1)
    state > length(iter) && return nothing
    return (iter[state], state + 1)
end

function add!(c::OpConnection, mel, cngs::StateChanges)
    i = findfirst(isequal(cngs), c.changes)
    isnothing(i) ? push!(c, (mel, cngs)) : c.mel[i] += mel
    return c
end

function add!(c::OpConnection, o::OpConnection)
    for (mel, cngs)=o
        add!(c, mel, cngs)
    end
end

Base.conj!(op::OpConnection) = conj!(op.mel)

function clear_duplicates(c::OpConnection)
    u_cngs = unique(c.changes)
    u_mel  = similar(c.mel, 0)
    for el=u_cngs
        ids = findall(isequal(el), c.changes)
        push!(u_mel, sum(c.mel[ids]))
    end
    return OpConnection(u_mel, u_cngs)
end
