"""
    time_evo_L <: AbstractProblem

Problem or finding the steady state of a ℒdagℒ matrix by computing
𝒞 = ∑|ρ(σ)|²|⟨⟨σ|ℒ |ρ⟩⟩|² using the sparse Liouvillian matrix.
"""
struct time_evo_L{B, SM} <: OpenTimeEvolutionProblem where {B<:Basis,
                                                 SM<:SparseMatrixCSC}
    HilbSpace::B            # 0
    L::SM
    ρss
end

time_evo_L(args...) = time_evo_L(STD_REAL_PREC, args...)
time_evo_L(T::Type{<:Number}, gl::GraphLindbladian) =
    time_evo_L(T, liouvillian(gl))
time_evo_L(T::Type{<:Number}, Liouv::SparseSuperOperator) =
    time_evo_L(first(Liouv.basis_l), sparse(transpose(Liouv.data)), 0.0)
time_evo_L(T::Type{<:Number}, Ham::DataOperator, cops::Vector) =
    time_evo_L(T, liouvillian(Ham, cops))

basis(prob::time_evo_L) = prob.HilbSpace

function compute_Cloc(prob::time_evo_L, net::MatrixNet, σ, lnψ=net(σ), σp=deepcopy(σ))
    ℒ = prob.L
    set_index!(σp, index(σ))

    C_loc = zero(Complex{real(out_type(net))})
    # Iterate through all elements in row i_σ of the matrix computing
    # ⟨i_σ|ℒdagℒψ⟩ = Σ_{i_σp} ⟨i_σ|ℒdagℒ|i_σp⟩⟨i_σp|ψ⟩
    # NOTE: ℒdagℒ is CSC, but I would like a CSR matrix. Since it is hermitian I
    # can simply take the conjugate of the elements in the columns
    i_σ = index(σ)
    for row_id = ℒ.colptr[i_σ]:(ℒ.colptr[i_σ+1]-1)
      # Find nonzero elements s by doing <i_sp|ℒdagℒ|i_σ>
      i_σp = ℒ.rowval[row_id]
      # BackConvert to int
      set_index!(σp, i_σp)
      # Compute the log(ψ(σ)/ψ(σ')), by only computing differences.
      log_ratio = logψ(net, σp) - lnψ
      # Conj because I am taking the transpose... see the note above.
      C_loc  += (ℒ.nzval[row_id]) * exp(log_ratio)
    end
    return C_loc
end
