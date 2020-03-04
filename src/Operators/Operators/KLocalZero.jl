"""
    KLocalOperatorZero

A simple KLocalOperator representing zero, but holding informations
about the hilbert basis.
"""
struct KLocalOperatorZero{H<:AbstractHilbert,T} <: AbsLinearOp{T}
    hilb::H
end

KLocalOperatorZero(T::Type{<:Number}, hilb) =
    KLocalOperatorZero{typeof(hilb), T}(hilb)

KLocalOperatorZero(hilb) = KLocalOperatorZero(ComplexF64, hilb)

LocalOperator(hilb) = KLocalOperatorZero(hilb)
LocalOperator(T, hilb) = KLocalOperatorZero(T, hilb)

export LocalOperator

sites(op::KLocalOperatorZero) = []

QuantumOpticsBase.basis(op::KLocalOperatorZero) = op.hilb

# Copy
duplicate(op::KLocalOperatorZero) = copy(op)
Base.copy(op::KLocalOperatorZero) = KLocalOperatorZero(basis(op))

##
function _row_valdiff!(conn::OpConnection, op::KLocalOperatorZero, v) end
function row_valdiff_index!(conn::OpConnectionIndex, op::KLocalOperatorZero, v) end
function accumulate_connections!(acc::AbstractAccumulator, op::KLocalOperatorZero, v) end
function map_connections(fun::Function , op::KLocalOperatorZero, v) end

Base.conj(op::KLocalOperatorZero) = copy(op)
Base.adjoint(op::KLocalOperatorZero) = copy(op)
Base.transpose(op::KLocalOperatorZero) = copy(op)

Base.:+(op_l::KLocalOperatorZero, op_r::KLocalOperator) = duplicate(op_r)
Base.:+(op_l::KLocalOperator, op_r::KLocalOperatorZero) = duplicate(op_l)

Base.:-(op_l::KLocalOperatorZero, op_r::KLocalOperator) = -op_r
Base.:-(op_l::KLocalOperator, op_r::KLocalOperatorZero) = -op_l

_sum_samesite(op_l::KLocalOperatorZero,
                op_r::AbsLinearOperator) = duplicate(opop_r_r)

_sum_samesite(op_l::AbsLinearOperator,
              op_r::KLocalOperatorZero) = duplicate(op_l)

Base.:*(a::Number, b::KLocalOperatorZero) = b

Base.:*(opl::KLocalOperatorZero, opr::KLocalOperator) =
    return opl
Base.:*(opl::KLocalOperator, opr::KLocalOperatorZero) =
    return opr

Base.:(==)(l::KLocalOperatorZero, r::KLocalOperatorZero) =
    l.hilb == r.hilb

Base.isapprox(l::KLocalOperatorZero, r::KLocalOperatorZero; kwargs...) =
    l.hilb == r.hilb

Base.show(io::IO, op::KLocalOperatorZero) = begin
    hilb   = basis(op)

    print(io, "KLocalOperator($hilb)")
end

Base.show(io::IO, m::MIME"text/plain", op::KLocalOperatorZero) = begin
    hilb   = basis(op)

    print(io, "empty KLocalOperator on space:  $hilb")
end
