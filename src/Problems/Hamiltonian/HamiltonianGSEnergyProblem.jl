struct HamiltonianGSEnergyProblem{B, SM} <: HermitianMatrixProblem where {B<:Basis,
                                                 SM}
    HilbSpace::B # 0
    H::SM
    ρss
end

HamiltonianGSEnergyProblem(args...) = HamiltonianGSEnergyProblem(STD_REAL_PREC, args...)
HamiltonianGSEnergyProblem(T::Type{<:Number}, gl::GraphOperator; operators=true) = begin
    if operators
        return HamiltonianGSEnergyProblem(basis(gl), to_linear_operator(gl), 0.0)
    else
        return HamiltonianGSEnergyProblem(T, SparseOperator(gl))
    end
end
HamiltonianGSEnergyProblem(T::Type{<:Number}, Ham::SparseOperator) =
    HamiltonianGSEnergyProblem(Ham.basis_l, data(Ham), 0.0)

basis(prob::HamiltonianGSEnergyProblem) = prob.HilbSpace

function compute_Cloc(prob::HamiltonianGSEnergyProblem{B,SM}, net::KetNet, σ::State,
                      lnψ=net(σ), σp=deepcopy(σ)) where {B,SM<:SparseMatrixCSC}
    H = prob.H

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
      log_ratio = logψ(net, σp) - lnψ
      # Conj because I am taking the transpose... see the note above.
      C_loc += conj(H.nzval[row_id]) * exp(log_ratio)
    end

    return C_loc
end

function compute_Cloc(prob::HamiltonianGSEnergyProblem{B,SM}, net::KetNet, σ::State,
                      lnψ=net(σ), σp=deepcopy(σ)) where {B,SM<:AbsLinearOperator}
    H = prob.H

    #### Now compute E(S) = Σₛ⟨s|H|ψ⟩/⟨s|ψ⟩
    C_loc = zero(Complex{real(out_type(net))})
    for op=operators(H)
        r = local_index(σ, sites(op))
        for (mel, changes)=op.op_conns[r]
            set_index!(σp, index(σ))
            for (site,val)=changes
                setat!(σp, site, val)
            end

            log_ratio = logψ(net, σp) - lnψ
            C_loc += mel * exp(log_ratio)
        end
    end

    return C_loc
end

# pretty printing
Base.show(io::IO, p::HamiltonianGSEnergyProblem) = print(io,
    """
    HamiltonianGSEnergyProblem: target minimum ground state energy
        - space : $(basis(p))
        - using operators : $(p.H isa AbsLinearOperator)""")
