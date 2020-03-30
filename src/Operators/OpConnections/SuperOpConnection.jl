struct SuperOpConnection{A,B,C} <: AbsOpConnection
    op_conn_l_id::A
    op_conn_r_id::B

    op_conn_l_r::C
end

SuperOpConnection{A,B,C}() where {A,B,C} = SuperOpConnection(A(), B(), C())

@inline length_l_id(c::SuperOpConnection) = length(c.op_conn_l_id)
@inline length_r_id(c::SuperOpConnection) = length(c.op_conn_r_id)
@inline length_l_r(c::SuperOpConnection)  = length(c.op_conn_l_r)
@inline Base.length(c::SuperOpConnection) = length_l_id(c) + length_r_id(c) + length_l_r(c)
@inline Base.size(c::SuperOpConnection) = (length(c), )
function Base.eltype(::Type{SuperOpConnection{L,R,LR}}) where {L,R,LR}
    T1, SC1 = eltype(L).parameters
    T2, SC2 = eltype(R).parameters
    T3, SC3 = eltype(R).parameters
    T = promote_type(T1, T2, T3)

    return Tuple{T, SC1}
end

Base.:(==)(a::SuperOpConnection, b::SuperOpConnection) = (a.op_conn_l_id == b.to_change &&
                                                          a.op_conn_r_id == b.op_conn_r_id &&
                                                          a.op_conn_l_r == b.op_conn_l_r )
# ad
#Base.@propagate_inbounds Base.getindex(c::OpConnection, i) = (c.mel[i], c.changes[i])

function Base.iterate(iter::SuperOpConnection, state=(1,(1,1)))

    el, inner_state = state

    if el == 1
        res = iterate(iter.op_conn_l_id, inner_state)

        if !isnothing(res)
            conn, inner = res
            return conn, (el, inner)
        else
            el = 2
            inner_state = (1,1)
        end
    end

    if el == 2
        res = iterate(iter.op_conn_r_id, inner_state)

        if !isnothing(res)
            conn, inner = res
            return conn, (el, inner)
        else
            el = 3
            inner_state = (1,1)
        end
    end

    if el == 3
        res = iterate(iter.op_conn_l_r, inner_state)

        if !isnothing(res)
            conn, inner = res
            return conn, (el, inner)
        else
            el = 4
            inner_state = (1,1)
        end
    end

    return nothing
end

@inline Base.getindex(c::SuperOpConnection, i) = begin
    if i <= length_l_id(c)
        return getindex(c.op_conn_l_id, i)
    elseif i <= length_l_id(c) + length_r_id(c)
        return getindex(c.op_conn_r_id, i - length_l_id(c))
    else
        return getindex(c.op_conn_l_r, i - length_l_id(c) - length_r_id(c))
    end
end

# showing
function Base.show(io::IO, c::SuperOpConnection)
    print(io, "$(length(c))-elements - SuperOpConnection{...}:\n")
    if length(c) > 0
        Base.print_matrix(IOContext(io, :compact=>true), collect(c))
    end
    return io
end
