struct HamiltonianGSVarianceProblem{B, SM} <: LRhoSquaredProblem where {B<:Basis,
                                                 SM}
    HilbSpace::B # 0
    H::SM
    ρss
end

HamiltonianGSVarianceProblem(args...) = HamiltonianGSVarianceProblem(STD_REAL_PREC, args...)
HamiltonianGSVarianceProblem(T::Type{<:Number}, gl::GraphOperator; operators=true) = begin
    if operators
        return HamiltonianGSVarianceProblem(basis(gl), to_linear_operator(gl), 0.0)
    else
        return HamiltonianGSVarianceProblem(T, SparseOperator(gl))
    end
end
HamiltonianGSVarianceProblem(T::Type{<:Number}, Ham::SparseOperator) =
    HamiltonianGSVarianceProblem(Ham.basis_l, data(Ham), 0.0)

basis(prob::HamiltonianGSVarianceProblem) = prob.HilbSpace

function compute_Cloc!(LLO_i, ∇lnψ, prob::HamiltonianGSVarianceProblem{B,SM}, net::KetNet, σ::State,
                      lnψ=net(σ), σp=deepcopy(σ)) where {B,SM<:SparseMatrixCSC}
    H = prob.H

    for el=LLO_i
      el .= 0.0
    end

    #### Now compute E(S) = Σₛ⟨s|Hψ⟩/⟨s|ψ⟩
    C_loc = zero(Complex{real(out_type(net))})
    # Iterate through all elements in row i_σ of the matrix computing
    # ⟨i_σ|H|ψ⟩ = Σ_{i_σp} ⟨i_σ|H|i_σp⟩⟨i_σp|ψ⟩
    # NOTE: H is CSC, but I would like a CSR matrix. Since it is hermitian I
    # can simply take the conjugate of the elements in the columns
    i_σ = index(σ)
    for row_id = H.colptr[i_σ]:(H.colptr[i_σ+1]-1)
      # Find nonzero elements s by doing <i_sp|ℒdagℒ|i_σ>
      i_σp = H.rowval[row_id]
      # BackConvert to int
      set_index!(σp, i_σp)
      # Compute the log(ψ(σ)/ψ(σ')), by only computing differences.
      lnψ_i, ∇lnψ_i = logψ_and_∇logψ!(∇lnψ, net, σp)
      C_loc_i = H.nzval[row_id] * exp(lnψ_i - lnψ) #TODO check

      C_loc  += C_loc_i
      for (LLOave, _∇lnψ)= zip(LLO_i, ∇lnψ_i.tuple_all_weights)
        LLOave .+= C_loc_i .* (_∇lnψ)
      end
    end

    return C_loc
end

function compute_Cloc!(∇𝒞σ, ∇lnψ, prob::HamiltonianGSVarianceProblem{B,SM}, net::KetNet, σ::State,
                      lnψσ=net(σ), σp=deepcopy(σ)) where {B,SM<:AbsLinearOperator}
    H = prob.H

    for ∇𝒞ᵢ=∇𝒞σ
      ∇𝒞ᵢ .= 0.0
    end

    #### Now compute E(S) = Σₛ⟨s|H|ψ⟩/⟨s|ψ⟩
    Cσ = zero(Complex{real(out_type(net))})
    for op=operators(H)
        r = local_index(σ, sites(op))
        for (mel, changes)=op.op_conns[r]
            set_index!(σp, index(σ))
            apply!(σp, changes)

            lnψ_σ̃ , ∇lnψ_σ̃ = logψ_and_∇logψ!(∇lnψ, net, σp)
            𝒞σ̃  =  mel * exp(lnψ_σ̃ - lnψσ)
            for (∇𝒞σᵢ, ∇lnψ_σ̃ᵢ)= zip(∇𝒞σ, ∇lnψ_σ̃.tuple_all_weights)
              ∇𝒞σᵢ .+= 𝒞σ̃ .* ∇lnψ_σ̃ᵢ
            end
            Cσ  += 𝒞σ̃
        end
    end

    return Cσ
end

# pretty printing
Base.show(io::IO, p::HamiltonianGSVarianceProblem) = print(io,
    """
    HamiltonianGSVarianceProblem: target minimum ground state energy
        - space : $(basis(p))
        - using operators : $(p.H isa AbsLinearOperator)""")
