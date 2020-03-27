struct StatesIterator{Safe,T,S}
    states::T
    sz::S
end

"""
    states(v::StateCollection)

Returns an iterator to iterate all states in the batch/array of batches
`v`.

By default, the iteration is done using unsafe views for increased performance.
"""
function states(v::Union{AbstractArray,AbstractDoubled}, safe=Val{false})
    sz = state_size(v)
    sz = CartesianIndices(sz)

    do_safe = safe isa Val{true} ? true : false
    return StatesIterator{do_safe, typeof(v), typeof(sz)}(v,sz)
end

Base.length(iter::StatesIterator) = length(iter.sz)
Base.eltype(iter::StatesIterator) = state_eltype(iter.states)
Base.lastindex(iter::StatesIterator) = length(iter)

function Base.iterate(it::StatesIterator{false}, state=1)
    state > length(it.sz) && return nothing

    return (state_uview(it.states, it.sz[state].I...), state+1)
end

function Base.getindex(it::StatesIterator{false}, i::Integer)
    @assert i>0 && i <= length(it)
    return state_uview(it.states, it.sz[i].I...)
end



####
struct BatchesIterator{Safe,T,S}
    states::T
    sz::S
    b_sz::Int  # size of batches
    b_num::Int # number of batches per entry
end

function batches(v::Union{AbstractArray,AbstractDoubled}, batch_size=batch_size(v), safe=Val{false})
    sz = state_size(v)
    batch_num, remainder = divrem(sz[1],batch_size)

    @assert remainder == 0

    sz = CartesianIndices(sz[2:end])
    do_safe = safe isa Val{true} ? true : false
    return BatchesIterator{do_safe, typeof(v), typeof(sz)}(v, sz, batch_size, batch_num)
end

Base.length(iter::BatchesIterator) = length(iter.sz)*iter.b_num
Base.eltype(iter::BatchesIterator) = state_eltype(iter.states)
Base.lastindex(iter::BatchesIterator) = length(iter)


function Base.iterate(it::BatchesIterator{false}, state=1)
    state > length(it.sz) && return nothing

    return (state_uview(it.states, it.sz[state].I...), state+1)
end

function Base.getindex(it::BatchesIterator{false}, i::Integer)
    @assert i>0 && i <= length(it)
    return state_uview(it.states, it.sz[i].I...)
end
