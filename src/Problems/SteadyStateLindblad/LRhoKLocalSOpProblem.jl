"""
    LRhoKLocalOpProblem <: AbstractProblem

Problem or finding the steady state of a ℒdagℒ matrix by computing
𝒞 = ∑|ρ(σ)|²|⟨⟨σ|ℒ |ρ⟩⟩|² using the sparse Liouvillian matrix.

DO NOT USE WITH COMPLEX-WEIGHT NETWORKS, AS IT DOES NOT WORK
"""
struct LRhoKLocalSOpProblem{LL} <: LRhoSquaredProblem
    L::LL
end

LRhoKLocalSOpProblem(gl::GraphLindbladian) = LRhoKLocalSOpProblem(STD_REAL_PREC, gl)
function LRhoKLocalSOpProblem(T, gl::GraphLindbladian)
    HnH, c_ops, c_ops_t = to_linear_operator(gl, Complex{real(T)})
    Liouv = KLocalLiouvillian(HnH, c_ops)
    return LRhoKLocalSOpProblem(Liouv)
end

QuantumOpticsBase.basis(prob::LRhoKLocalSOpProblem) = basis(prob.L)
operator(prob::LRhoKLocalSOpProblem) = prob.L

# pretty printing
Base.show(io::IO, p::LRhoKLocalSOpProblem) = print(io,
    "LRhoKLocalSOpProblem on space $(basis(p)) computing the variance of Lrho using the sparse liouvillian")
