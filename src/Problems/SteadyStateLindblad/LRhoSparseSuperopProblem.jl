"""
    LRhoSparseSuperopProblem <: AbstractProblem

Problem or finding the steady state of a ℒdagℒ matrix by computing
𝒞 = ∑|ρ(σ)|²|⟨⟨σ|ℒ |ρ⟩⟩|² using the sparse Liouvillian matrix.

DO NOT USE WITH COMPLEX-WEIGHT NETWORKS, AS IT DOES NOT WORK
"""
struct LRhoSparseSuperopProblem{B, SM} <: LRhoSquaredProblem where {B<:Basis,
                                                 SM<:SparseMatrixCSC}
    HilbSpace::B            # 0
    L::SM
    ρss
end

"""
    LRhoSparseSuperopProblem([T=STD_REAL_PREC], args...)

Creates a problem for minimizing the cost function 𝒞 = ∑|ρ(σ)|²|⟨⟨σ|ℒ |ρ⟩⟩|².
Computes |⟨⟨σ|ℒ |ρ⟩⟩| by building the sparse superoperator, which can be done
for sizes up to dimℋ < 500.

For more than 9 spins it is reccomended to use the command `LRhoSparseOpProblem`.

args... can either be a `GraphLindbladian`, or the Hamiltonian and a vector
of collapse operators.
"""
LRhoSparseSuperopProblem(args...) = LRhoSparseSuperopProblem(STD_REAL_PREC, args...)
LRhoSparseSuperopProblem(T::Type{<:Number}, gl::GraphLindbladian) =
    LRhoSparseSuperopProblem(T, liouvillian(gl))
LRhoSparseSuperopProblem(T::Type{<:Number}, Liouv::SparseSuperOperator) =
    LRhoSparseSuperopProblem(first(Liouv.basis_l), sparse(transpose(Liouv.data)), 0.0)
LRhoSparseSuperopProblem(T::Type{<:Number}, Ham::DataOperator, cops::Vector) =
    LRhoSparseSuperopProblem(T, liouvillian(Ham, cops))

basis(prob::LRhoSparseSuperopProblem) = prob.HilbSpace

function compute_Cloc!(LLO_i, ∇lnψ, prob::LRhoSparseSuperopProblem, net::MatrixNet,
                       σ, lnψ=net(σ), σp=deepcopy(σ))
    ℒ = prob.L
    set_index!(σp, index(σ))

    for el=LLO_i
      el .= 0.0
    end

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
      lnψ_i, ∇lnψ_i = logψ_and_∇logψ!(∇lnψ, net, σp)
      # Conj because I am taking the transpose... see the note above.
      C_loc_i = (ℒ.nzval[row_id]) * exp(lnψ_i - lnψ) #TODO check

      C_loc  += C_loc_i
      for (LLOave, _∇lnψ)= zip(LLO_i, ∇lnψ_i.tuple_all_weights)
        LLOave .+= C_loc_i .* (_∇lnψ)
      end
    end
    return C_loc
end

# pretty printing
Base.show(io::IO, p::LRhoSparseSuperopProblem) = print(io,
    "LRhoSparseSuperopProblem on space $(basis(p)) computing the variance of Lrho using the sparse liouvillian")
