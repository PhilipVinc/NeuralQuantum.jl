"""
    LdagL_Lrho_op_prob <: Problem

Problem or finding the steady state of a ℒdagℒ matrix by computing
𝒞 = ∑|ρ(σ)|²|⟨⟨σ|ℒ |ρ⟩⟩|² using the sparse Liouvillian matrix.

DO NOT USE WITH COMPLEX-WEIGHT NETWORKS, AS IT DOES NOT WORK
"""
struct LdagL_Lrho_op_prob{B, SM1, SM} <: LRhoSquaredProblem where {B<:Basis,
                                                 SM<:SparseMatrixCSC}
    HilbSpace::B            # 0
    HnH::SM1
    L_ops::Vector{SM}       # 4
    L_ops_t::Vector{SM}     # 5
    ρss
end

LdagL_Lrho_op_prob(gl::GraphLindbladian) =
    LdagL_Lrho_op_prob(Float64, gl)

function LdagL_Lrho_op_prob(T, gl::GraphLindbladian)

    HnH, c_ops, c_ops_t = to_linear_operator(gl)

    return LdagL_Lrho_op_prob(basis(gl), HnH, c_ops, c_ops_t, 0.0)
end

basis(prob::LdagL_Lrho_op_prob) = prob.HilbSpace

# Standard method dispatched when the state is generic (non lut).
# will work only if 𝝝 and 𝝝p are the same type (and non lut!)
function compute_Cloc!(LLO_i, ∇lnψ, prob::LdagL_Lrho_op_prob,
                       net::MatrixNet, 𝝝::S,
                       lnψ=net(𝝝), 𝝝p::S=deepcopy(𝝝)) where {S}
    # hey
    HnH = prob.HnH
    c_ops = prob.L_ops
    c_ops_trans = prob.L_ops_t

    set_index!(𝝝p, index(𝝝))
    𝝝p_row = row(𝝝p)
    𝝝p_col = col(𝝝p)

    for el=LLO_i
      el .= 0.0
    end

    C_loc = zero(Complex{real(out_type(net))})

    # ⟨σ|Hρ|σt⟩ (using hermitianity of HdH)
    diffs_hnh = row_valdiff(HnH, row(𝝝))
    set_index!(𝝝p_col, index(col(𝝝)))
    for (mel, changes)=diffs_hnh

        set_index!(𝝝p_row, index(row(𝝝)))
        for (site,val)=changes
            setat!(𝝝p_row, site, val)
        end

        lnψ_i, ∇lnψ_i = logψ_and_∇logψ!(∇lnψ, net, 𝝝p)
        C_loc_i  =  -1.0im * mel * exp(lnψ_i - lnψ)
        for (LLOave, _∇lnψ)= zip(LLO_i, ∇lnψ_i.tuple_all_weights)
          LLOave .+= C_loc_i .* _∇lnψ
        end
        C_loc  += C_loc_i
    end

    # ⟨σ|ρHᴴ|σt⟩
    diffs_hnh = row_valdiff(HnH, col(𝝝))
    set_index!(𝝝p_row, index(row(𝝝)))
    for (mel, changes)=diffs_hnh
        set_index!(𝝝p_col, index(col(𝝝)))
        for (site,val)=changes
            setat!(𝝝p_col, site, val)
        end

        lnψ_i, ∇lnψ_i = logψ_and_∇logψ!(∇lnψ, net, 𝝝p)
        C_loc_i  =  1.0im * conj(mel) * exp(lnψ_i - lnψ)
        for (LLOave, _∇lnψ)= zip(LLO_i, ∇lnψ_i.tuple_all_weights)
          LLOave .+= C_loc_i .* _∇lnψ
        end
        C_loc  += C_loc_i
    end

    # L rho Ldag H #ok
    # -im ⟨σ|L ρ Lᴴ|σt⟩
    for L=c_ops
        diffs_r = row_valdiff(L, row(𝝝)) # TODO Not allocate!
        diffs_c = row_valdiff(L, col(𝝝))

        for (mel_r, changes_r)=diffs_r
            set_index!(𝝝p_row, index(row(𝝝)))
            for (site,val)=changes_r
                setat!(𝝝p_row, site, val)
            end

            for (mel_c, changes_c)=diffs_c
                set_index!(𝝝p_col, index(col(𝝝)))
                for (site,val)=changes_c
                    setat!(𝝝p_col, site, val)
                end

                lnψ_i, ∇lnψ_i = logψ_and_∇logψ!(∇lnψ, net, 𝝝p)
                C_loc_i  =  (mel_r) * conj(mel_c) *  exp(lnψ_i - lnψ)

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
Base.show(io::IO, p::LdagL_Lrho_op_prob) = print(io,
    "LdagL_Lrho_op_prob on space $(basis(p)) computing the variance of Lrho using the sparse liouvillian")

# Variant for when the state has a LookUpTable and resorts to computing
# only lut updates.
function compute_Cloc!(LLO_i, ∇lnψ, prob::LdagL_Lrho_op_prob,
                       net::MatrixNet, 𝝝::S,
                       _lnψ=nothing, _𝝝p::NS=nothing) where {S<:LUState, NS<:Union{Nothing, S}}
    # hey
    HnH = prob.HnH
    c_ops = prob.L_ops
    c_ops_trans = prob.L_ops_t

    for el=LLO_i
      el .= 0.0
    end
    𝝝s = state(𝝝)
    no_changes = changes(row(𝝝s))

    C_loc = zero(Complex{real(out_type(net))})

    # ⟨σ|Hρ|σt⟩ (using hermitianity of HdH)
    # TODO should be non allocating!
    diffs_hnh = row_valdiff(HnH, raw_state(row(𝝝s)))
    for (mel, changes)=diffs_hnh
        Δ_lnψ, ∇lnψ_i = Δ_logψ_and_∇logψ!(∇lnψ, net, 𝝝, changes, no_changes)

        C_loc_i  =  -1.0im * mel * exp(Δ_lnψ)
        for (LLOave, _∇lnψ)= zip(LLO_i, ∇lnψ_i.tuple_all_weights)
          LLOave .+= C_loc_i .* _∇lnψ
        end
        C_loc  += C_loc_i
    end

    # ⟨σ|ρHᴴ|σt⟩
    resize!(diffs_hnh, 0)
    row_valdiff!(diffs_hnh, HnH, raw_state(col(𝝝s)))
    for (mel, changes)=diffs_hnh
        Δ_lnψ, ∇lnψ_i = Δ_logψ_and_∇logψ!(∇lnψ, net, 𝝝, no_changes, changes)

        C_loc_i  =  1.0im * conj(mel) * exp(Δ_lnψ)
        for (LLOave, _∇lnψ)= zip(LLO_i, ∇lnψ_i.tuple_all_weights)
          LLOave .+= C_loc_i .* _∇lnψ
        end
        C_loc  += C_loc_i
    end

    # L rho Ldag H #ok
    # -im ⟨σ|L ρ Lᴴ|σt⟩
    for L=c_ops
        diffs_r = row_valdiff(L, raw_state(row(𝝝s)))
        diffs_c = row_valdiff(L, raw_state(col(𝝝s)))

        for (mel_r, changes_r)=diffs_r

            for (mel_c, changes_c)=diffs_c
                Δ_lnψ, ∇lnψ_i = Δ_logψ_and_∇logψ!(∇lnψ, net, 𝝝, changes_r, changes_c)

                C_loc_i  =  (mel_r) * conj(mel_c) *  exp(Δ_lnψ)
                for (LLOave, _∇lnψ)= zip(LLO_i, ∇lnψ_i.tuple_all_weights)
                  LLOave .+= C_loc_i .* _∇lnψ
                end
                C_loc  += C_loc_i
            end
        end
    end

    return C_loc
end
