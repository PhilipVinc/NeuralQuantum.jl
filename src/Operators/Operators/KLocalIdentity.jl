struct KLocalIdentity{T,H} <: AbsLinearOp{T}
    hilb::H
end

KLocalIdentity(T::Type, hilb::AbstractHilbert) =
    KLocalIdentity{T,typeof(hilb)}(hilb)

KLocalIdentity(hilb) = KLocalIdentity(ComplexF64, hilb)

QuantumOpticsBase.basis(op::KLocalIdentity) = op.hilb

Base.copy(op::KLocalIdentity{T,H}) where {T,H} = KLocalIdentity{T,H}(basis(op))
duplicate(op::KLocalIdentity) = copy(op)

conn_type(top::Type{KLocalIdentity{T,H}}) where {T,H} =
    OpConnectionIdentity{Vector{Int}, Vector{T}}
