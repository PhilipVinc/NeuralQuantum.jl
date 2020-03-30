"""
    OpConnectionIdentity

An iterable structure containing all the changes from one state to another, and
an associated value.
This effectively encodes a full row of an operator.
"""
struct OpConnection{A<:AbstractArray,B<:AbstractArray,C<:AbstractArray} <: AbsOpConnection
    mel::A
    changes::Vector{StateChanges{B,C}}

    length::Ref{Int}
end

function OpConnection(op::AbsLinearOperator)
    t = conn_type(op)
    v1, v2, v3 = t.parameters
    return OpConnection(v1(), Vector{StateChanges{v2,v3}}(), Ref(0))
end

function OpConnection{A,B,C}() where {A,B,C}
    return OpConnection(A(), Vector{typeof(StateChanges(B(),C()))}(), Ref(0))
end

function Base.resize!(c::OpConnection, n)
    # diminish size
    if n <= length(c)
        c.length[] = n
    else
        # increase size
        c.length[] = n
        resize!(c.mel,     n)
        resize!(c.changes, n)
    end

    return c
end

Base.eltype(::Type{OpConnection{A,B,C}}) where {A,B,C} = Tuple{eltype(A), StateChanges{B,C}}
Base.length(c::OpConnection) = c.length[]
Base.size(c::OpConnection) = (length(c), )
capacity(c::OpConnection) = length(c.mel)


Base.:(==)(a::OpConnection, b::OpConnection) = (a.mel == b.mel &&
                                              a.changes == b.changes)

@inline Base.push!(c::OpConnection, (m_els, to_change, new_values)) =
    push!(c, (m_els, StateChanges(to_change, new_values)))

function Base.push!(c::OpConnection, (mel, cngs)::Tuple{Number, StateChanges})
    if length(c) < capacity(c)
        resize!(c, length(c)+1)
        @inbounds setindex!(c, length(c), (mel, cngs))
    else
        c.length[] += 1
        push!(c.mel, mel)
        push!(c.changes, cngs)
    end
    return c
end

@inline function Base.setindex!(c::OpConnection, i, (mel, cngs)::Tuple{Number, StateChanges})
    @boundscheck checkbounds(1:length(c), i)
    @inbounds c.mel[i]     = mel
    @inbounds c.changes[i] = cngs
end

function Base.append!(c::OpConnection, (m_els, to_change, new_values)::Tuple{Vector, Vector, Vector})
    cngs = Vector{StateChanges{eltype(to_change), eltype(new_values)}}()
    for (tcng, nwvls) = zip(to_change, new_values)
        push!(cngs, StateChanges(tcng, nwvls))
    end
    return append!(c, (m_els, cngs))
end

function Base.append!(c::OpConnection, (m_els, cngs)::Tuple)
    l = length(c)
    resize!(c, l+length(m_els))

    for i=1:length(m_els)
        @inbounds setindex!(c, l+i, (m_els[i], cngs[i]))
    end
    return c
end

function Base.append!(c::OpConnection, new::OpConnection)
    L = length(new)

    L == 0 && return c

    # If this accumulator is empty, then simply append
    if length(c) == 0
        append!(c, (new.mel, new.changes))
    else
        # We assume that the first element is always for len(statechanges) = 0
        c.mel[1] += new.mel[1]

        L == 1 && return c

        append!(c, (uview(new.mel, 2:L), uview(new.changes, 2:L )))
    end

    return c
end

Base.@propagate_inbounds Base.getindex(c::OpConnection, i) = (c.mel[i], c.changes[i])

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

#showing
function Base.show(io::IO, c::OpConnection{A,B,C}) where {A,B,C}
    print(io, "$(length(c))-elements - OpConnection{$(eltype(A)),StateChanges{$B,$C}}:\n")
    Base.print_matrix(IOContext(io, :compact=>true), collect(c))
    return io
end
