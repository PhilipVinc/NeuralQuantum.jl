"""
    KLocalOperatorTensor

A KLocalOperator representing the sum of several KLocalOperator-s. Internally,
the sum is stored as a vector of local operators acting on some sites.
"""
struct KLocalOperatorTensor{H,T,O1,O2} <: AbsLinearSuperOperator{T}
    hilb::H
    sites::T

    # list of sites in this sum
    op_l::O1
    op_r::O2
end

"""
    Builds the tensor product of two operators
"""
function KLocalOperatorTensor(op_l, op_r)
    if op_l isa KLocalIdentity
        st = (Int[],       sites(op_r))
        hilb = basis(op_r)
    elseif op_r isa KLocalIdentity
        st = (sites(op_l), Int[])
        hilb = basis(op_l)
    else
        @assert basis(op_l) == basis(op_r)
        hilb = basis(op_l)
        st = (sites(op_l), sites(op_r))
    end

    KLocalOperatorTensor(SuperOpSpace(hilb), st, op_l, op_r)
end

function KLocalOperatorTensor(op_l::KLocalOperatorSum, op_r::KLocalIdentity)
    ops = [KLocalOperatorTensor(op, op_r) for op=operators(op_l)]
    return sum(ops)
end

function KLocalOperatorTensor(op_l::KLocalIdentity, op_r::KLocalOperatorSum)
    ops = [KLocalOperatorTensor(op_l, op) for op=operators(op_r)]
    return sum(ops)
end

KLocalOperatorSum(op::KLocalOperatorTensor) = KLocalOperatorSum(basis(op), [sites(op)], [op])

function KLocalOperatorTensor(op_l::KLocalOperatorSum, op_r::KLocalOperatorSum)
    throw("error not impl")
end

QuantumOpticsBase.basis(op::KLocalOperatorTensor) = op.hilb
sites(op::KLocalOperatorTensor) = op.sites


## Accessors
operators(op::KLocalOperatorTensor) = (op,)

conn_type(op::KLocalOperatorTensor{H,T,O1,O2}) where {H,T,O1,O2}      = conn_type(op)
#conn_type(op::KLocalOperatorTensor{H,T,O1,Nothing}) where {H,T,O1}    = conn_type(op.op_l)
#conn_type(op::KLocalOperatorTensor{H,T,Nothing,O2}) where {H,T,O2}    = conn_type(op.op_r)

conn_type(::Type{KLocalOperatorTensor{H,T,O1,O2}}) where {H,T,O1,O2}  =
    OpConnectionTensor{conn_type(O1), conn_type(O2)}
conn_type(::Type{KLocalOperatorTensor{H,T,O1,Nothing}}) where{H,T,O1} = #conn_type(O1)
    OpConnectionTensor{conn_type(O1), eye_conn_type(O1)}
conn_type(::Type{KLocalOperatorTensor{H,T,Nothing,O2}}) where{H,T,O2} = #conn_type(O2)
    OpConnectionTensor{eye_conn_type(O2), conn_type(O2)}
conn_type(::Type{KLocalOperatorTensor{H,T,O1,<:KLocalIdentity}}) where{H,T,O1} = #conn_type(O1)
    OpConnectionTensor{conn_type(O1), eye_conn_type(O1)}
conn_type(::Type{KLocalOperatorTensor{H,T,<:KLocalIdentity,O2}}) where{H,T,O2} = #conn_type(O2)
    OpConnectionTensor{eye_conn_type(O2), conn_type(O2)}

duplicate(::Nothing) = nothing

function duplicate(op::KLocalOperatorTensor)
    KLocalOperatorTensor(duplicate(op.op_l), duplicate(op.op_r))
end

function _row_valdiff!(conn::AbsOpConnection, op::KLocalOperatorTensor, v::ADoubleState)
    op_r = op.op_r
    op_l = op.op_l
    if op_r isa KLocalIdentity
        r_r = local_index(row(v), basis(op_l), sites(op_l))
        append!(conn, (op_l.op_conns[r_r], nothing))
    elseif op_l isa KLocalIdentity
        r_c = local_index(col(v), basis(op_r), sites(op_r))
        append!(conn, (nothing, op_r.op_conns[r_c]))
    else
        r_r = local_index(row(v), basis(op_l), sites(op_l))
        r_c = local_index(col(v), basis(op_r), sites(op_r))
        append!(conn, (op_l.op_conns[r_r], op_r.op_conns[r_c]))
    end
    return conn
end


function map_connections(fun::Function, op::KLocalOperatorTensor, v::ADoubleState)
    op_r = op.op_r
    op_l = op.op_l
    hilb_ph = physical(basis(op))
    if op_r isa KLocalIdentity
        r = local_index(row(v), hilb_ph, sites(op_l))

        for (mel,changes)=op_l.op_conns[r]
            #fun(mel, 1.0, changes, nothing, v)
            fun(mel, (changes, nothing), v)
        end
    elseif op_l isa KLocalIdentity
        r = local_index(col(v), hilb_ph, sites(op_r))

        for (mel,changes)=op_r.op_conns[r]
            #fun(1.0, mel, nothing, changes, v)
            fun(mel, (nothing, changes), v)
        end
    else
        r_r = local_index(row(v), hilb_ph, sites(op_l))
        r_c = local_index(col(v), hilb_ph, sites(op_r))

        for (mel_r, changes_r)=op_l.op_conns[r_r]
            for (mel_c, changes_c)=op_r.op_conns[r_c]
                #fun(mel_r, mel_c, changes_r, changes_c, v)
                fun(mel_r*mel_c, (changes_r, changes_c), v)
            end
        end
    end
    return nothing
end

function accumulate_connections!(acc::AbstractAccumulator, op::KLocalOperatorTensor, v::ADoubleState)
    op_l = op.op_l; op_r = op.op_r

    if op.op_r isa KLocalIdentity
        r = local_index(row(v), basis(op_l), sites(op_l))

        for (mel,changes)=op_l.op_conns[r]
            #fun(mel, 1.0, changes, nothing, v)
            acc(mel, changes, nothing, v)
        end
    elseif op.op_l isa KLocalIdentity
        r = local_index(col(v), basis(op_r), sites(op_r))

        for (mel,changes)=op_r.op_conns[r]
            #fun(1.0, mel, nothing, changes, v)
            acc(mel, nothing, changes, v)
        end
    else
        r_r = local_index(row(v), basis(op_l), sites(op_l))
        r_c = local_index(col(v), basis(op_r), sites(op_r))

        for (mel_r, changes_r)=op_l.op_conns[r_r]
            for (mel_c, changes_c)=op_r.op_conns[r_c]
                acc(mel_r*mel_c, changes_r, changes_c, v)
            end
        end
    end
    return acc
end


_add_samesite(op_l::KLocalOperatorTensor, op_r::KLocalOperatorTensor) = _add_samesite!(duplicate(op_l), op_r)

_add_samesite!(::Nothing, ::Nothing) = nothing
function _add_samesite!(op_l::KLocalOperatorTensor, op_r::KLocalOperatorTensor)
    @assert op_l.sites == op_r.sites

    _add_samesite!(op_l.op_l, op_r.op_l)
    _add_samesite!(op_l.op_r, op_r.op_r)
    return op_l
end

function Base.conj!(op::KLocalOperatorTensor)
    !isnothing(op.op_l) && conj!(op.op_l)
    !isnothing(op.op_r) && conj!(op.op_r)
    return op
end

function Base.transpose(op::KLocalOperatorTensor)
    KLocalOperatorTensor(op.op_r, op.op_l)

end

Base.conj(op::KLocalOperatorTensor) = conj!(duplicate(op))

Base.adjoint(op::KLocalOperatorTensor) = conj(transpose(op))


Base.:+(op_l::KLocalOperatorTensor, op_r::KLocalOperatorTensor) = begin
    sites(op_l) == sites(op_r) && return _add_samesite(op_l, op_r)
    return KLocalOperatorSum(op_l) + op_r
end

Base.:*(a::Number, b::KLocalOperatorTensor) =
    _op_alpha_prod(b,a)

function _op_alpha_prod(op::KLocalOperatorTensor, a::Number)
    if op.op_l isa KLocalIdentity
        return KLocalOperatorTensor(op.op_l, a*op.op_r)
    elseif op.op_r isa KLocalIdentity
        return KLocalOperatorTensor(a*op.op_l, op.op_r)
    else
        a = sqrt(a)
        return KLocalOperatorTensor(a*op.op_l, a*op.op_r)
    end
end

Base.isapprox(l::KLocalOperatorTensor, r::KLocalOperatorTensor; kwargs...) =
    isapprox(l.op_l, r.op_l; kwargs...) && isapprox(l.op_r, r.op_r; kwargs...)

Base.:(==)(l::KLocalOperatorTensor, r::KLocalOperatorTensor) =
    l.op_l == r.op_l && l.op_r == r.op_r

Base.eltype(::T) where {T<:KLocalOperatorTensor} = eltype(T)
Base.eltype(T::Type{<:KLocalOperatorTensor{A,B,C}}) where {A,B,C<:KLocalOperator} =
    eltype(C)
Base.eltype(T::Type{<:KLocalOperatorTensor{A,B,C,D}}) where {A,B,C,D<:KLocalOperator} =
    eltype(D)
Base.eltype(T::Type{<:KLocalOperatorTensor{A,B,C,D}}) where {A,B,C<:KLocalOperator,D<:KLocalOperator} =
    eltype(C)
