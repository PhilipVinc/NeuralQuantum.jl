"""
    OpConnectionIdentity

An OpConnection to store the operatorial identity.
It always returns one entry with value `true` and no changes. 
"""
struct OpConnectionIdentity{B,C}
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
