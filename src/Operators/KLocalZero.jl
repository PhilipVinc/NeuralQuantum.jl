"""
    KLocalOperatorZero

A simple KLocalOperator representing zero, but holding informations
about the hilbert basis.
"""
struct KLocalOperatorZero{H<:AbstractBasis} <: AbsLinearOperator
    hilb::H
end

LocalOperator(hilb) = KLocalOperatorZero(hilb)
export LocalOperator

sites(op::KLocalOperatorZero) = []

QuantumOpticsBase.basis(op::KLocalOperatorZero) = op.hilb

# Copy
duplicate(op::KLocalOperatorZero) = KLocalOperatorZero(basis(op))

##
function row_valdiff!(conn::OpConnection, op::KLocalOperatorZero, v::State) end
function row_valdiff_index!(conn::OpConnectionIndex, op::KLocalOperatorZero, v::State) end
function accumulate_connections!(acc::AbstractAccumulator, op::KLocalOperatorZero, v::State) end
function map_connections(fun::Function , op::KLocalOperatorZero, v) end

Base.:+(op_l::KLocalOperatorZero, op_r::KLocalOperator) = duplicate(op_r)
Base.:+(op_l::KLocalOperator, op_r::KLocalOperatorZero) = duplicate(op_l)

_sum_samesite(op_l::KLocalOperatorZero,
                op_r::AbsLinearOperator) = duplicate(opop_r_r)

_sum_samesite(op_l::AbsLinearOperator,
              op_r::KLocalOperatorZero) = duplicate(op_l)

Base.:*(a::Number, b::KLocalOperatorZero) = b

Base.:*(opl::KLocalOperatorZero, opr::KLocalOperator) =
    return opl
Base.:*(opl::KLocalOperator, opr::KLocalOperatorZero) =
    return opr

Base.show(io::IO, op::KLocalOperatorZero) = begin
    hilb   = basis(op)

    print(io, "KLocalOperator($hilb)")
end

Base.show(io::IO, m::MIME"text/plain", op::KLocalOperatorZero) = begin
    hilb   = basis(op)

    print(io, "empty KLocalOperator on space:  $hilb")
end
