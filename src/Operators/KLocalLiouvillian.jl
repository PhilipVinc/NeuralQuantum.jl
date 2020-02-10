struct KLocalLiouvillian{H,T,A,B,C} <: AbsLinearOperator
    hilb::H
    sites::T
    HnH_l::A
    HnH_r::B
    LLdag::C
end

function KLocalLiouvillian(HnH, Lops)
    hilb = basis(HnH)
    T = eltype(HnH)
    HnH_l = T(-1.0im) * KLocalOperatorTensor(HnH, nothing)
    HnH_r = T(1.0im) * KLocalOperatorTensor(nothing, HnH')

    LLdag_list = [KLocalOperatorTensor(L, conj(L)) for L=Lops]
    LLdag = isempty(LLdag_list) ? LocalOperator(SuperOpSpace(basis(HnH))) : sum(LLdag_list)

    return KLocalLiouvillian(SuperOpSpace(basis(HnH)),[], HnH_l, HnH_r, LLdag)
end

"""
    liouvillian(Ĥ, ops::Vector)

Constructs the LocalOperator representation of a liouvillian super-operator with
coherent part given by the hamilotnian `Ĥ`, and `ops` as the list of jump operators.
"""
function QuantumOpticsBase.liouvillian(H::AbsLinearOperator, Lops::AbstractVector)
    HnH = duplicate(H)
    for L=Lops
        HnH += -0.5im * L'*L
    end
    return KLocalLiouvillian(HnH, Lops)
end

function QuantumOpticsBase.liouvillian(Lops::AbstractVector)
    hilb = basis(first(Lops))
    liouvillian(LocalOperator(hilb), Lops)
end
sites(op::KLocalLiouvillian) = op.sites
QuantumOpticsBase.basis(op::KLocalLiouvillian) = op.hilb

conn_type(op::KLocalLiouvillian) = conn_type(op.HnH_l)

accumulate_connections!(a, b::Vector, c) = nothing

function accumulate_connections!(acc::AbstractAccumulator, op::KLocalLiouvillian, v::ADoubleState)
    accumulate_connections!(acc, op.HnH_l, v)
    accumulate_connections!(acc, op.HnH_r, v)
    accumulate_connections!(acc, op.LLdag, v)

    return acc
end

function _row_valdiff!(conn::OpConnection, op::KLocalLiouvillian, v::ADoubleState)
    _row_valdiff!(conn, op.HnH_l, v)
    _row_valdiff!(conn, op.HnH_r, v)
    _row_valdiff!(conn, op.LLdag, v)
    return conn
end

function map_connections(fun::Function, op::KLocalLiouvillian, v::ADoubleState)
    map_connections(fun, op.HnH_l, v)
    map_connections(fun, op.HnH_r, v)
    map_connections(fun, op.LLdag, v)
    return nothing
end

Base.eltype(::T) where {T<:KLocalLiouvillian} = eltype(T)
Base.eltype(t::Type{<:KLocalLiouvillian{H,T,A,B,C}}) where {H,T,A,B,C} =
    eltype(A)

Base.show(io::IO, m::MIME"text/plain", op::KLocalLiouvillian) = begin
    T    = eltype(op.HnH_l)
    h    = basis(op)

    print(io, "KLocalLiouvillian($T)\n  Hilb: $h")
end
