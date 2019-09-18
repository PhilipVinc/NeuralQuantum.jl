"""
    LdagLSparseSuperopProblem <: AbstractProblem

Problem or finding the steady state of a ℒdagℒ matrix
"""
struct LdagLSparseSuperopProblem{B, SM} <: HermitianMatrixProblem where {B<:Basis,
                                                 SM<:SparseMatrixCSC}
    HilbSpace::B            # 0
    LdagL::SM
    ρss
end

"""
    LdagLSparseSuperopProblem([T=STD_REAL_PREC], lindbladian)

Creates a problem for minimizing the cost function 𝒞 = ∑|ρ(σ)|²|⟨⟨σ|ℒ'ℒ |ρ⟩⟩|².
Computes |⟨⟨σ|ℒ'ℒ |ρ⟩⟩| by building the sparse superoperator, which can be done
for sizes up to dimℋ < 500. For more than 9 spins it is reccomended to use the
command LRhoSparseOpProblem

`lindbladian` can either be the lindbladian on a graph, a QuantumOptics superoperator
or the Hamiltonian and a vector of collapse operators.

`T=STD_REAL_PREC` by default is the numerical precision used. It should match that of
the network.
"""
LdagLSparseSuperopProblem(args...) = LdagLSparseSuperopProblem(STD_REAL_PREC, args...)
LdagLSparseSuperopProblem(T::Type{<:Number}, gl::GraphLindbladian) =
    LdagLSparseSuperopProblem(T, liouvillian(gl))
LdagLSparseSuperopProblem(T::Type{<:Number}, Ham::DataOperator, cops::Vector) =
    LdagLSparseSuperopProblem(T, liouvillian(Ham, cops))
LdagLSparseSuperopProblem(T::Type{<:Number}, Liouv::SparseSuperOperator) =
    LdagLSparseSuperopProblem(first(Liouv.basis_l), Liouv.data'*Liouv.data, 0.0)

basis(prob::LdagLSparseSuperopProblem) = prob.HilbSpace

function compute_Cloc(prob::LdagLSparseSuperopProblem, net::MatrixNet, σ, lnψ=net(σ), σp=deepcopy(σ))
    ℒdagℒ = prob.LdagL
    set_index!(σp, index(σ))

    #### Now compute E(S) = Σₛ⟨s|Hψ⟩/⟨s|ψ⟩
    C_loc = zero(Complex{real(out_type(net))})
    # Iterate through all elements in row i_σ of the matrix computing
    # ⟨i_σ|ℒdagℒψ⟩ = Σ_{i_σp} ⟨i_σ|ℒdagℒ|i_σp⟩⟨i_σp|ψ⟩
    # NOTE: ℒdagℒ is CSC, but I would like a CSR matrix. Since it is hermitian I
    # can simply take the conjugate of the elements in the columns
    i_σ = index(σ)
    for row_id = ℒdagℒ.colptr[i_σ]:(ℒdagℒ.colptr[i_σ+1]-1)
      # Find nonzero elements s by doing <i_sp|ℒdagℒ|i_σ>
      i_σp = ℒdagℒ.rowval[row_id]
      # BackConvert to int
      set_index!(σp, i_σp)
      # Compute the log(ψ(σ)/ψ(σ')), by only computing differences.
      log_ratio = logψ(net, σp) - lnψ
      # Conj because I am taking the transpose... see the note above.
      C_loc += conj(ℒdagℒ.nzval[row_id]) * exp(log_ratio)
    end
    C_loc
end

Base.show(io::IO, p::LdagLSparseSuperopProblem) = print(io,
    "LdagLSparseSuperopProblem on space : $(basis(p)), computing the energy of LdagL with the sparse liouvillian")
