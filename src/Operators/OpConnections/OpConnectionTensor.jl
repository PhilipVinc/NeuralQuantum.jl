struct OpConnectionTensor{L,R} <: AbsOpConnection
    conn_l::L
    conn_r::R
end

OpConnectionTensor{L,R}() where {L,R} = OpConnectionTensor(L(), R())

@inline length_l(c::OpConnectionTensor) = length(c.conn_l)
@inline length_r(c::OpConnectionTensor) = length(c.conn_r)
@inline Base.length(c::OpConnectionTensor) = length_l(c) * length_r(c)
@inline Base.size(c::OpConnectionTensor) = (length(c), )


Base.:(==)(a::OpConnectionTensor, b::OpConnectionTensor) = (a.conn_l == b.conn_l &&
                                                            a.conn_r == b.conn_r)
#a
function Base.resize!(c::OpConnectionTensor, i)
    resize!(c.conn_l, i)
    resize!(c.conn_r, i)
    return c
end

function Base.append!(c::OpConnectionTensor, (conn_l, conn_r))
    append!(c.conn_l, conn_l)
    append!(c.conn_r, conn_r)
    return c
end

function Base.iterate(iter::OpConnectionTensor, state=(1,1))
    state_l, state_r = state

    if state_r == length(iter.conn_r) + 1
        state_r = 1
        state_l += 1
    end

    if state_l == length(iter.conn_l) + 1
        return nothing
    end

    mel_l, changes_l = iter.conn_l[state_l]
    mel_r, changes_r = iter.conn_r[state_r]

    return (mel_l*mel_r, (changes_l, changes_r)), (state_l, state_r+1)
end

function Base.getindex(c::OpConnectionTensor, i)
    l,r = divrem(i-1, length_r(c))
    return first(iterate(c, (l+1, r+1)))
end
