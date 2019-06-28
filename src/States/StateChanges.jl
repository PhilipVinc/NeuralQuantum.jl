
abstract type AbstractStateChanges end

struct StateChanges{A<:AbstractArray,B<:AbstractArray} <: AbstractStateChanges
    to_change::A
    new_values::B
end

function StateChanges(state::State)
    return StateChanges(Int[], eltype(state)[])
end

function Base.resize!(c::StateChanges, n)
    resize!(c.to_change,  n)
    resize!(c.new_values, n)
end

zero!(c::StateChanges) = resize!(c, 0)

Base.eltype(c::StateChanges{A,B}) where {A,B} = (eltype(A), eltype(B))
Base.length(c::StateChanges) = length(c.to_change)
Base.size(c::StateChanges) = size(c.to_change)

function Base.:(==)(a::StateChanges, b::StateChanges)
    sk_a = sortperm(a.to_change)
    sk_b = sortperm(b.to_change)

    return ( a.to_change[sk_a]  == b.to_change[sk_b] &&
             a.new_values[sk_a] == b.new_values[sk_b] )
end

function Base.push!(c::StateChanges, (to_change, new_values))
    push!(c.to_change, to_change)
    push!(c.new_values, new_values)
    return c
end

function Base.append!(c::StateChanges, (to_change, new_values))
    append!(c.to_change, to_change)
    append!(c.new_values, new_values)
    return c
end

Base.getindex(c::StateChanges, i) = (c.to_change[i], c.new_values[i])

function Base.iterate(iter::StateChanges, state=1)
    state > length(iter) && return nothing
    return (iter[state], state + 1)
end

function Base.show(io::IO, ch::StateChanges)
    print(io, "StateChanges{$(eltype(ch))} : [")
    content = ""
    for (id, val)=ch
        content *= "$id->$val, "
    end
    print(io, content[1:end-2]*"]")
end
