"""
    LdagLSparseOpProblem <: AbstractProblem

Problem or finding the steady state of a ℒdagℒ matrix
"""
struct LdagLSparseOpProblem{B, SM} <: HermitianMatrixProblem where {B<:Basis,
                                                 SM<:SparseMatrixCSC}
    HilbSpace::B            # 0
    H::SM                   # 1
    H_t::SM                 # 2
    HdH::SM                 # 3
    L_ops::Vector{SM}       # 4
    L_ops_t::Vector{SM}     # 5

    LdH_ops::Vector{SM}     # 6
    HdL_ops::Vector{SM}     # 7
    LdL_ops_t::Matrix{SM}   # 8
    ρss
end

basis(prob::LdagLSparseOpProblem) = prob.HilbSpace


"""
    LdagLSparseOpProblem([T=STD_REAL_PREC], lindbladian)

Creates a problem for minimizing the cost function 𝒞 = ∑|ρ(σ)|²|⟨⟨σ|ℒ'ℒ |ρ⟩⟩|².
Computes |⟨⟨σ|ℒ'ℒ |ρ⟩⟩| by computing on the fly commutators with the
Hamiltonian and with the collapse operators.

`lindbladian` can either be the lindbladian on a graph, a QuantumOptics superoperator
or the Hamiltonian and a vector of collapse operators.

`T=STD_REAL_PREC` by default is the numerical precision used. It should match that of
the network.
"""
LdagLSparseOpProblem(args...) = LdagLSparseOpProblem(STD_REAL_PREC, args...)
LdagLSparseOpProblem(T::Type{<:Number}, gl::GraphLindbladian) =
    LdagLSparseOpProblem(T, basis(gl), SparseOperator(hamiltonian(gl)), jump_operators(gl))
LdagLSparseOpProblem(T::Type{<:Number}, ham::DataOperator, cops) =
    LdagLSparseOpProblem(T, ham.basis_l, ham, cops)
function LdagLSparseOpProblem(T::Type{<:Number}, Hilb::Basis, Ham::DataOperator, c_ops_q)
    # Fix complex numbers
    if real(T) == T
        T = Complex{T}
    end
    # Generate H_eff
    H_eff   = deepcopy(T.(Ham.data))
    ST = typeof(H_eff)

    c_ops       = Vector{ST}(undef, length(c_ops_q))
    c_ops_trans = Vector{ST}(undef, length(c_ops_q))
    for i=1:length(c_ops)
        c_ops[i]       = c_ops_q[i].data
        c_ops_trans[i] = transpose(c_ops[i])
        H_eff         -= 0.5im * (c_ops[i]'*c_ops[i])
    end

    LdH_ops     = Vector{ST}(undef, length(c_ops))
    HdL_ops     = Vector{ST}(undef, length(c_ops))
    for i=1:length(c_ops)
        LdH_ops[i]     = c_ops[i]'*H_eff
        HdL_ops[i]     = LdH_ops[i]'
    end

    LdL_ops_t   = Matrix{ST}(undef, length(c_ops), length(c_ops))
    for i=1:length(c_ops)
        for j=1:length(c_ops)
            LdL_ops_t[i,j] = transpose(c_ops[i]'*c_ops[j])
        end
    end

    LdagLSparseOpProblem{typeof(Hilb), ST}(
                 Hilb,                  # 0
                 H_eff,                 # 1
                 transpose(H_eff),
                 H_eff'*H_eff,          # 3
                 c_ops,
                 c_ops_trans,           # 5
                 LdH_ops,
                 HdL_ops,               # 7
                 LdL_ops_t,             # 8
                 0.0)
end

function compute_Cloc(prob::LdagLSparseOpProblem, net::MatrixNet, 𝝝, lnψ=net(𝝝), 𝝝p=deepcopy(𝝝))
    # Quantities of the problem
    H = prob.H
    H_t = prob.H_t
    HdH = prob.HdH
    c_ops = prob.L_ops
    c_ops_trans = prob.L_ops_t
    LdH_ops = prob.LdH_ops
    HdL_ops = prob.HdL_ops
    LdL_ops_t = prob.LdL_ops_t

    σ  = row(𝝝)
    σt = col(𝝝)
    set_index!(𝝝p, index(𝝝))
    𝝝p_row = row(𝝝p)
    𝝝p_col = col(𝝝p)

    i_σt = index(σt)
    i_σ  = index(σ)

    C_loc = zero(Complex{real(out_type(net))})

    # ⟨σ|ρHᴴH|σt⟩
    set!(𝝝p_row, toint(σ))
    for row_id = HdH.colptr[i_σt]:(HdH.colptr[i_σt+1]-1)
      i_σ_p = HdH.rowval[row_id]
      set_index!(𝝝p_col, i_σ_p)

      log_ratio = net(𝝝p) - lnψ
      ΔE  =  HdH.nzval[row_id] * exp(log_ratio) #mat[i_σ, i_σ_p]
      C_loc  += ΔE #2.0*real(ΔE)
    end

    # ⟨σ|HᴴHρ|σt⟩ (using hermitianity of HdH)
    set!(𝝝p_col, toint(σt))
    for row_id = HdH.colptr[i_σ]:(HdH.colptr[i_σ+1]-1)
      i_σ_p = HdH.rowval[row_id]
      set_index!(𝝝p_row, i_σ_p)

      log_ratio = logψ(net, 𝝝p) - lnψ
      ΔE  =  conj(HdH.nzval[row_id]) * exp(log_ratio) #mat[i_σ_p, i_σt]
      C_loc  += ΔE #2.0*real(ΔE)
    end

    # ⟨σ|HᴴρHᴴ|σt⟩
    for L_row_id = H.colptr[i_σ]:(H.colptr[i_σ+1]-1)
      i_σ_p = H.rowval[L_row_id]
      L_σp_σ = H.nzval[L_row_id]
      set_index!(𝝝p_row, i_σ_p)

      for Ldag_row_id = H_t.colptr[i_σt]:(H_t.colptr[i_σt+1]-1)
        # Find nonzero elements s by doing <i_σp|L|i_σ> and backconvert
        i_σt_p = H_t.rowval[Ldag_row_id]
        set_index!(𝝝p_col, i_σt_p)

        log_ratio = logψ(net, 𝝝p)- lnψ
        #@assert (mat[i_σ_p, i_σt_p]/exp(lnψ) - log_ratio) == 0
        ΔE  =  - conj(L_σp_σ) * conj(H_t.nzval[Ldag_row_id]) * exp(log_ratio) # mat[i_σ_p, i_σt_p]
        C_loc  += ΔE #2.0*real(ΔE)
      end
    end


    # ⟨σ|HρH|σt⟩
    for L_row_id = H_t.colptr[i_σ]:(H_t.colptr[i_σ+1]-1)
      i_σ_p = H_t.rowval[L_row_id]
      L_σp_σ = H_t.nzval[L_row_id]
      set_index!(𝝝p_row, i_σ_p)

      for Ldag_row_id = H.colptr[i_σt]:(H.colptr[i_σt+1]-1)
        # Find nonzero elements s by doing <i_σp|L|i_σ> and backconvert
        i_σt_p = H.rowval[Ldag_row_id]
        set_index!(𝝝p_col, i_σt_p)

        log_ratio = logψ(net, 𝝝p) - lnψ

        ΔE  =  - L_σp_σ * H.nzval[Ldag_row_id] * exp(log_ratio) # mat[i_σ_p, i_σt_p]
        C_loc  += ΔE #2.0*real(ΔE)
      end
    end


    # L rho Ldag H #ok
    # -im ⟨σ|L ρ LᴴH|σt⟩
    for i=1:length(LdH_ops)
      LdH = LdH_ops[i]
      L   = c_ops_trans[i]

      for ext_row_id = L.colptr[i_σ]:(L.colptr[i_σ+1]-1)
        i_σ_p = L.rowval[ext_row_id]
        set_index!(𝝝p_row, i_σ_p)
        val_σ_p = L.nzval[ext_row_id]

        for int_row_id = LdH.colptr[i_σt]:(LdH.colptr[i_σt+1]-1)
          i_σt_p = LdH.rowval[int_row_id]
          set_index!(𝝝p_col, i_σt_p)

          log_ratio = logψ(net, 𝝝p) - lnψ
          #@assert (mat[i_σ_p, i_σt_p] - log_ratio) == 0
          ΔE  = -1.0im * val_σ_p * LdH.nzval[int_row_id] *  exp(log_ratio) # mat[i_σ_p, i_σt_p]
          C_loc  += ΔE # 2.0*real(ΔE)
        end
      end
    end

    # +im ⟨σ|HᴴL ρ Lᴴ|σt⟩
    for i=1:length(LdH_ops)
      LdH = LdH_ops[i]
      L   = c_ops_trans[i]

      for ext_row_id = L.colptr[i_σt]:(L.colptr[i_σt+1]-1)
        i_σ_p = L.rowval[ext_row_id]
        set_index!(𝝝p_col, i_σ_p)
        val_σ_p = L.nzval[ext_row_id]

        for int_row_id = LdH.colptr[i_σ]:(LdH.colptr[i_σ+1]-1)
          i_σt_p = LdH.rowval[int_row_id]
          set_index!(𝝝p_row, i_σt_p)

          log_ratio = logψ(net, 𝝝p) - lnψ
          #@assert (mat[i_σt_p, i_σ_p] - log_ratio) == 0

          ΔE  = 1.0im * conj(val_σ_p) * conj(LdH.nzval[int_row_id]) *  exp(log_ratio) # mat[i_σt_p, i_σ_p]
          C_loc  += ΔE # 2.0*real(ΔE)
        end
      end
    end


    # Ldag rho Hdag L #ok
    # ⟨σ|Lᴴ ρ HᴴL|σt⟩
    for i=1:length(LdH_ops)
      HdL = HdL_ops[i]
      L   = c_ops[i]

      for ext_row_id = L.colptr[i_σ]:(L.colptr[i_σ+1]-1)
        i_σ_p = L.rowval[ext_row_id]
        set_index!(𝝝p_row, i_σ_p)
        val_σ_p = L.nzval[ext_row_id]

        for int_row_id = HdL.colptr[i_σt]:(HdL.colptr[i_σt+1]-1)
          i_σt_p = HdL.rowval[int_row_id]
          set_index!(𝝝p_col, i_σt_p)

          log_ratio = logψ(net, 𝝝p) - lnψ
          #@assert (mat[i_σ_p, i_σt_p] - log_ratio) == 0

          ΔE  = 1.0im * conj(val_σ_p) * HdL.nzval[int_row_id] * exp(log_ratio) # mat[i_σ_p, i_σt_p]
          C_loc  += ΔE # 2.0*real(ΔE)
        end
      end
    end

    # ⟨σ|Lᴴ H ρ L|σt⟩
    for i=1:length(LdH_ops)
      HdL = HdL_ops[i]
      L   = c_ops[i]

      for ext_row_id = L.colptr[i_σt]:(L.colptr[i_σt+1]-1)
        i_σ_p = L.rowval[ext_row_id]
        set_index!(𝝝p_col, i_σ_p)
        val_σ_p = L.nzval[ext_row_id]

        for int_row_id = HdL.colptr[i_σ]:(HdL.colptr[i_σ+1]-1)
          i_σt_p = HdL.rowval[int_row_id]
          set_index!(𝝝p_row, i_σt_p)

          log_ratio = logψ(net, 𝝝p) - lnψ
          #@assert (mat[i_σt_p, i_σ_p] - log_ratio) == 0

          ΔE  = -1.0im * val_σ_p * conj(HdL.nzval[int_row_id]) * exp(log_ratio) # mat[i_σt_p, i_σ_p]
          C_loc  += ΔE # 2.0*real(ΔE)
        end
      end
    end

    for LdL=LdL_ops_t
      for ext_row_id = LdL.colptr[i_σ]:(LdL.colptr[i_σ+1]-1)
        i_σ_p = LdL.rowval[ext_row_id]
        set_index!(𝝝p_row, i_σ_p)
        val_σ_p = LdL.nzval[ext_row_id]
        for int_row_id = LdL.colptr[i_σt]:(LdL.colptr[i_σt+1]-1)
          i_σt_p = LdL.rowval[int_row_id]
          set_index!(𝝝p_col, i_σt_p)

          log_ratio = logψ(net, 𝝝p) - lnψ
          #@assert (mat[i_σ_p, i_σt_p] - log_ratio) == 0

          ΔE  = val_σ_p * conj(LdL.nzval[int_row_id]) * exp(log_ratio) #mat[i_σ_p, i_σt_p]
          C_loc  += ΔE
        end
      end
    end

    return C_loc
end

Base.show(io::IO, p::LdagLSparseOpProblem) = print(io,
    "LdagLSparseOpProblem on space : $(basis(p)), computing the energy of LdagL with H, jump_ops")
