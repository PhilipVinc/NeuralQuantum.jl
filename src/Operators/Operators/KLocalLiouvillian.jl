struct KLocalLiouvillian{H,T,Vs,A<:AbsLinearSuperOperator{T},B,C} <: AbsLinearSuperOperator{T}
    hilb::H
    sites::Vs
    HnH_l::A
    HnH_r::B
    LLdag::C
end

function KLocalLiouvillian(HnH, Lops)
    hilb = basis(HnH)
    T = eltype(HnH)
    HnH_l = T(-1.0im) * KLocalOperatorTensor(HnH, KLocalIdentity(T, hilb))
    HnH_r = T(1.0im) * KLocalOperatorTensor(KLocalIdentity(T, hilb), HnH')

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

conn_type(op::KLocalLiouvillian) = conn_type(typeof(op))
conn_type(::Type{KLocalLiouvillian{H,T,Vs,A,B,C}}) where {H,T,Vs,A,B,C}  =
    SuperOpConnection{conn_type(A), conn_type(B), conn_type(C)}

function accumulate_connections!(acc::AbstractAccumulator, op::KLocalLiouvillian, v)
    accumulate_connections!(acc, op.HnH_l, v)
    accumulate_connections!(acc, op.HnH_r, v)
    accumulate_connections!(acc, op.LLdag, v)

    return acc
end

function row_valdiff!(conn::SuperOpConnection, op::KLocalLiouvillian, v::ADoubleState; init=true)
    row_valdiff!(conn.op_conn_l_id, op.HnH_l, v; init=init)
    row_valdiff!(conn.op_conn_r_id, op.HnH_r, v; init=init)
    row_valdiff!(conn.op_conn_l_r,  op.LLdag, v; init=init)

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

function Base.show(io::IO, m::MIME"text/plain", op::KLocalLiouvillian)
    T    = eltype(op.HnH_l)
    h    = basis(op)

    print(io, "KLocalLiouvillian($T)\n\t- Hilb: $h")
end

function Base.show(io::IO, op::KLocalLiouvillian)
    T    = eltype(op.HnH_l)
    h    = basis(op)

    print(io, "KLocalLiouvillian($T) on Hilb: $h")
end
