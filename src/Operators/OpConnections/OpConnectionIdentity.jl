"""
    OpConnectionIdentity

An OpConnection to store the operatorial identity.
It always returns one entry with value `true` and no changes.
"""
struct OpConnectionIdentity{B,C} <: AbsOpConnection
    c::StateChanges{B,C}
end

OpConnectionIdentity(a::Type, b::Type) =
    OpConnectionIdentity(StateChanges{a,b}())

OpConnectionIdentity{B,C}() where {B,C} = OpConnectionIdentity{B,C}(StateChanges{B,C}())

function OpConnectionIdentity(op::AbsLinearOperator)
    t = conn_type(op)
    v1, v2, v3 = t.parameters
    return OpConnectionIdentity(v2, v3)
end

@inline Base.length(c::OpConnectionIdentity) = true
@inline Base.resize!(c::OpConnectionIdentity, i) = c
@inline Base.append!(c::OpConnectionIdentity, a::Nothing) = c
@inline Base.eltype(c::OpConnectionIdentity{B,C}) where {B,C} =
    Tuple{Bool, StateChanges{B,C}}

Base.:(==)(::OpConnectionIdentity, ::OpConnectionIdentity) = true

@inline function Base.iterate(iter::OpConnectionIdentity, state=(1))
    if state == 1
        return (true, iter.c)
    else
        return nothing
    end
end

@inline function Base.getindex(c::OpConnectionIdentity, i)
    @assert i == 1
    return (true, c.c)
end

#showing
function Base.show(io::IO, c::OpConnectionIdentity{B,C}) where {B,C}
    print(io, "$(length(c))-elements - Identity-OpConnection{bool,StateChanges{$B,$C}}:\n")
    print(io, " (true, [])")
    return io
end
