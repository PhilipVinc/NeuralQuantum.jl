"""
    KLocalOperatorSum

A KLocalOperator representing the sum of several KLocalOperator-s. Internally,
the sum is stored as a vector of local operators acting on some sites.
"""
struct KLocalOperatorSum{VS<:AbstractVector,VOp} <: AbsLinearOperator
    # list of sites in this sum
    sites::VS

    # List of operators
    operators::VOp
end

KLocalOperatorSum(op::KLocalOperator) = KLocalOperatorSum([sites(op)], [op])

## Accessors
operators(op::KLocalOperatorSum) = op.operators
conn_type(op::KLocalOperatorSum) = conn_type(eltype(operators(op)))


# Copy
function duplicate(op::KLocalOperatorSum)
    new_sites = deepcopy(op.sites)
    ops = eltype(op.operators)[];
    for _op=operators(op)
        push!(ops, duplicate(_op))
    end

    KLocalOperatorSum(new_sites, ops)
end

#
function row_valdiff!(conn::OpConnection, op::KLocalOperatorSum, v::State)
    for _op=operators(op)
        row_valdiff!(conn, _op, v)
    end
    conn
end

function row_valdiff_index!(conn::OpConnectionIndex, op::KLocalOperatorSum, v::State)
    for _op=operators(op)
        row_valdiff_index!(conn, _op, v)
    end
    conn
end

function Base.sum!(op_sum::KLocalOperatorSum, op::KLocalOperator)
    id = findfirst(isequal(sites(op)), op_sum.sites)

    if isnothing(id)
        push!(op_sum.sites, sites(op))
        push!(op_sum.operators, op)
    else
        sum_samesite!(op_sum.operators[id], op)
    end

    return op_sum
end

function Base.sum!(op_l::KLocalOperatorSum, op_r::KLocalOperatorSum)
    for op=operators(op_r)
        sum!(op_l, op)
    end
    op_l
end

+(op_l::KLocalOperatorSum, op::KLocalOperator) = sum!(duplicate(op_l), op)
+(op::KLocalOperator, ops::KLocalOperatorSum) = ops + op

function Base.transpose(ops::KLocalOperatorSum)
    new_sites  = similar(ops.sites)
    new_sites .= ops.sites

    new_ops    = eltype(operators(ops))[]
    for op = operators(ops)
        push!(new_ops, transpose(op))
    end
    return KLocalOperatorSum(new_sites, new_ops)
end

function Base.conj(ops::KLocalOperatorSum)
    new_sites  = similar(ops.sites)
    new_sites .= ops.sites

    new_ops    = eltype(operators(ops))[]
    for op = operators(ops)
        push!(new_ops, conj(op))
    end
    return KLocalOperatorSum(new_sites, new_ops)
end

function Base.conj!(ops::KLocalOperatorSum)
    for op = operators(ops)
        conj!(op)
    end
    return ops
end

Base.adjoint(ops::KLocalOperatorSum) = conj!(transpose(ops))

Base.show(io::IO, ::MIME"text/plain", op::KLocalOperatorSum) = print(io,
    "KLocalOperatorSum: \n\t -sites: $(op.sites)")

Base.eltype(::T) where {T<:KLocalOperatorSum} = eltype(T)
Base.eltype(T::Type{KLocalOperatorSum{Vec,VOp}}) where {Vec,VOp} =
    eltype(eltype(VOp))
