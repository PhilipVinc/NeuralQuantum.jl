"""
    KLocalOperatorSum

A KLocalOperator representing the sum of several KLocalOperator-s. Internally,
the sum is stored as a vector of local operators acting on some sites.
"""
struct KLocalOperatorSum{H<:AbstractHilbert, VS<:AbstractVector,VOp} <: AbsLinearOperator
    hilb::H

    # list of sites in this sum
    sites::VS

    # List of operators
    operators::VOp
end

KLocalOperatorSum(op::KLocalOperator) = KLocalOperatorSum(basis(op), [sites(op)], [op])
KLocalOperatorSum(op::AbsLinearOperator) = KLocalOperatorSum(basis(op), [sites(op)], [op])
KLocalOperatorSum(op::KLocalOperatorSum) = duplicate(op)

## Accessors
QuantumOpticsBase.basis(op::KLocalOperatorSum) = op.hilb
operators(op::KLocalOperatorSum) = op.operators
conn_type(op::KLocalOperatorSum) = conn_type(eltype(operators(op)))
conn_type(op::Type{KLocalOperatorSum{H,Vs,VOp}}) where {H,Vs,VOp} = conn_type(eltype(VOp))
sites(op::KLocalOperatorSum) = op.sites

# Copy
function duplicate(op::KLocalOperatorSum)
    new_sites = deepcopy(op.sites)
    ops = eltype(op.operators)[];
    for _op=operators(op)
        push!(ops, duplicate(_op))
    end

    return KLocalOperatorSum(basis(op), new_sites, ops)
end

#
function _row_valdiff!(conn::AbsOpConnection, op::KLocalOperatorSum, v)
    for _op=operators(op)
        _row_valdiff!(conn, _op, v)
    end
    return conn
end

function row_valdiff_index!(conn::OpConnectionIndex, op::KLocalOperatorSum, v)
    for _op=operators(op)
        row_valdiff_index!(conn, _op, v)
    end
    return conn
end

function map_connections(fun::Function, ∑Ô::KLocalOperatorSum, v)
    for Ô=operators(∑Ô)
        map_connections(fun, Ô, v)
    end
    return nothing
end

function accumulate_connections!(acc::AbstractAccumulator, ∑Ô::KLocalOperatorSum, v)
    for Ô=operators(∑Ô)
        accumulate_connections!(acc, Ô, v)
    end
    return acc
end

function _add!(op_sum::KLocalOperatorSum, op::AbsLinearOperator)
    id = findfirst(isequal(sites(op)), op_sum.sites)

    if isnothing(id)
        push!(op_sum.sites, sites(op))
        push!(op_sum.operators, op)
    else
        _add_samesite!(op_sum.operators[id], op)
    end

    return op_sum
end

function _add!(op_l::KLocalOperatorSum, op_r::KLocalOperatorSum)
    for op=operators(op_r)
        _add!(op_l, op)
    end
    op_l
end

Base.:+(op::AbsLinearOperator, ops::KLocalOperatorSum) = ops + op
Base.:-(op::AbsLinearOperator, ops::KLocalOperatorSum) = ops - op

Base.:+(op_l::KLocalOperatorSum, op::AbsLinearOperator) = _add!(duplicate(op_l), op)
Base.:-(op_l::KLocalOperatorSum, op::AbsLinearOperator) = _add!(duplicate(op_l), -op)

Base.:+(op_l::KLocalOperatorSum, op::KLocalOperatorSum) = _add!(duplicate(op_l), op)
Base.:-(op_l::KLocalOperatorSum, op::KLocalOperatorSum) = _add!(duplicate(op_l), -op)

Base.:+(op_l::KLocalOperator, op_r::KLocalOperator) = begin
    sites(op_l) == sites(op_r) && return _add_samesite(op_l, op_r)
    return KLocalOperatorSum(op_l) + op_r
end

Base.:-(op_l::KLocalOperator, op_r::KLocalOperator) = begin
    sites(op_l) == sites(op_r) && return _add_samesite(op_l, -op_r)
    return KLocalOperatorSum(op_l) - op_r
end


function Base.transpose(ops::KLocalOperatorSum)
    new_sites  = copy(ops.sites)

    new_ops    = eltype(operators(ops))[]
    for op = operators(ops)
        push!(new_ops, transpose(op))
    end
    return KLocalOperatorSum(basis(ops), new_sites, new_ops)
end

function Base.conj(ops::KLocalOperatorSum)
    new_sites  = similar(ops.sites)
    new_sites .= ops.sites

    new_ops    = eltype(operators(ops))[]
    for op = operators(ops)
        push!(new_ops, conj(op))
    end
    return KLocalOperatorSum(basis(ops), new_sites, new_ops)
end

function Base.conj!(ops::KLocalOperatorSum)
    for op = operators(ops)
        conj!(op)
    end
    return ops
end

Base.adjoint(ops::KLocalOperatorSum) = conj!(transpose(ops))

function Base.:*(opl::KLocalOperatorSum, opr::KLocalOperator)
    ∑op =  duplicate(opl)
    for (i,op)=enumerate(operators(∑op))
        op_new = op*opr
        ∑op.operators[i] = op_new
        ∑op.sites[i] = sites(op_new)
    end
    return ∑op
end

function Base.:*(opl::KLocalOperator, opr::KLocalOperatorSum)
    ∑op =  duplicate(opr)
    for (i,op)=enumerate(operators(∑op))
        op_new = opl*op
        ∑op.operators[i] = op_new
        ∑op.sites[i] = sites(op_new)
    end
    return ∑op
end

Base.show(io::IO, ::MIME"text/plain", op::KLocalOperatorSum) = print(io,
    "KLocalOperatorSum: \n\t -sites: $(op.sites)")

Base.show(io::IO, op::KLocalOperatorSum) = print(io,
    "KLocalOperatorSum: on sites: $(op.sites)")

Base.eltype(::T) where {T<:KLocalOperatorSum} = eltype(T)
Base.eltype(T::Type{KLocalOperatorSum{H,Vec,VOp}}) where {H,Vec,VOp} =
    eltype(eltype(VOp))

Base.:*(a::Number, b::KLocalOperatorSum) =
    _op_alpha_prod(b,a)

function _op_alpha_prod(ops::KLocalOperatorSum, a::Number)
    op_all = [a*op for op=operators(ops)]
    return sum(op_all)
end
