"""
    LRhoSparseOpProblem <: AbstractProblem

Problem or finding the steady state of a ℒdagℒ matrix by computing
𝒞 = ∑|ρ(σ)|²|⟨⟨σ|ℒ |ρ⟩⟩|² only storing H and c_ops.

DO NOT USE WITH COMPLEX-WEIGHT NETWORKS, AS IT DOES NOT WORK
"""
struct LRhoSparseOpProblem{B, SM} <: LRhoSquaredProblem where {B<:Basis,
                                                 SM<:SparseMatrixCSC}
    HilbSpace::B            # 0
    HnH::SM
    HnH_t::SM
    L_ops::Vector{SM}       # 4
    L_ops_h::Vector{SM}     # 4
    L_ops_t::Vector{SM}     # 5
    ρss
end

"""
    LRhoSparseOpProblem([T=STD_REAL_PREC], args...)

Creates a problem for minimizing the cost function 𝒞 = ∑|ρ(σ)|²|⟨⟨σ|ℒ |ρ⟩⟩|².
Computes |⟨⟨σ|ℒ |ρ⟩⟩| by computing on the fly the commutator with the
Hamiltonian and with the collapse operators.

`args...` can either be a `GraphLindbladian`, or the Hamiltonian and a vector
of collapse operators.

`T=STD_REAL_PREC` by default is the numerical precision used. It should match that of
the network.
"""
LRhoSparseOpProblem(args...) = LRhoSparseOpProblem(STD_REAL_PREC, args...)
LRhoSparseOpProblem(T::Type{<:Number}, gl::GraphLindbladian) =
    LRhoSparseOpProblem(T, basis(gl), SparseOperator(hamiltonian(gl)), jump_operators(gl))
LRhoSparseOpProblem(T::Type{<:Number}, Ham::DataOperator, cops::Vector) =
    LRhoSparseOpProblem(T, Ham.basis_l, Ham, cops)
function LRhoSparseOpProblem(T::Type{<:Number}, Hilb::Basis, Ham::DataOperator, c_ops_q::Vector)
    # Fix complex numbers
    if real(T) == T
        T = Complex{T}
    end

    # Generate H_eff
    H_eff   = deepcopy(T.(Ham.data))
    ST = typeof(H_eff)

    c_ops       = Vector{ST}(undef, length(c_ops_q))
    c_ops_h     = Vector{ST}(undef, length(c_ops_q))
    c_ops_trans = Vector{ST}(undef, length(c_ops_q))
    for i=1:length(c_ops)
        c_ops[i]       = c_ops_q[i].data
        c_ops_h[i]     = c_ops[i]'
        c_ops_trans[i] = transpose(c_ops[i])
        H_eff         -= 0.5im * (c_ops[i]'*c_ops[i])
    end

    LRhoSparseOpProblem{typeof(Hilb), ST}(Hilb,                  # 0
                    H_eff,
                    transpose(H_eff),
                    c_ops,
                    c_ops_h,
                    c_ops_trans,
                    0.0)
end

basis(prob::LRhoSparseOpProblem) = prob.HilbSpace

function compute_Cloc!(LLO_i, ∇lnψ, prob::LRhoSparseOpProblem, net::MatrixNet, 𝝝,
                      lnψ=net(𝝝), 𝝝p=deepcopy(𝝝))
    HnH = prob.HnH
    HnH_t = prob.HnH_t
    c_ops = prob.L_ops
    c_ops_h = prob.L_ops_h
    c_ops_trans = prob.L_ops_t

    σ  = row(𝝝)
    σt = col(𝝝)
    set_index!(𝝝p, index(𝝝))
    𝝝p_row = row(𝝝p)
    𝝝p_col = col(𝝝p)

    for el=LLO_i
      el .= 0.0
    end

    i_σt = index(σt)
    i_σ  = index(σ)

    C_loc = zero(Complex{real(out_type(net))})

    # ⟨σ|Hρ|σt⟩ (using hermitianity of HdH)
    set!(𝝝p_col, toint(σt))
    for row_id = HnH_t.colptr[i_σ]:(HnH_t.colptr[i_σ+1]-1)
      i_σ_p = HnH_t.rowval[row_id]
      set_index!(𝝝p_row, i_σ_p)
      lnψ_i, ∇lnψ_i = logψ_and_∇logψ!(∇lnψ, net, 𝝝p)
      C_loc_i  =  -1.0im * HnH_t.nzval[row_id] * exp(lnψ_i - lnψ)

      for (LLOave, _∇lnψ)= zip(LLO_i, ∇lnψ_i.tuple_all_weights)
        LLOave .+= C_loc_i .* _∇lnψ
      end
      C_loc  += C_loc_i
     end

    # ⟨σ|ρHᴴ|σt⟩
    set!(𝝝p_row, toint(σ))
    for row_id = HnH.colptr[i_σt]:(HnH.colptr[i_σt+1]-1)
      i_σ_p = HnH.rowval[row_id]
      set_index!(𝝝p_col, i_σ_p)
      lnψ_i, ∇lnψ_i = logψ_and_∇logψ!(∇lnψ, net, 𝝝p)
      C_loc_i  =  1.0im * conj(HnH_t.nzval[row_id]) * exp(lnψ_i - lnψ)

      for (LLOave, _∇lnψ)= zip(LLO_i, ∇lnψ_i.tuple_all_weights)
        LLOave .+= C_loc_i .* _∇lnψ
      end
      C_loc  += C_loc_i
    end

    # L rho Ldag H #ok
    # -im ⟨σ|L ρ Lᴴ|σt⟩
    for i=1:length(c_ops_h)
      Ld  = c_ops_h[i]
      L   = c_ops_trans[i]

      for ext_row_id = L.colptr[i_σ]:(L.colptr[i_σ+1]-1)
        i_σ_p = L.rowval[ext_row_id]
        set_index!(𝝝p_row, i_σ_p)
        val_σ_p = L.nzval[ext_row_id]

        for int_row_id = Ld.colptr[i_σt]:(Ld.colptr[i_σt+1]-1)
          i_σt_p = Ld.rowval[int_row_id]
          set_index!(𝝝p_col, i_σt_p)

          lnψ_i, ∇lnψ_i = logψ_and_∇logψ!(∇lnψ, net, 𝝝p)
          C_loc_i  =  val_σ_p * Ld.nzval[int_row_id] *  exp(lnψ_i - lnψ)

          for (LLOave, _∇lnψ)= zip(LLO_i, ∇lnψ_i.tuple_all_weights)
            LLOave .+= C_loc_i .* _∇lnψ
          end
          C_loc  += C_loc_i
        end
      end
    end

    return C_loc
end

# pretty printing
Base.show(io::IO, p::LRhoSparseOpProblem) = print(io,
    "LRhoSparseOpProblem on space : $(basis(p)) computing the variance of Lrho using sparse H, c_ops")
