export SR, sr_none, sr_multiplicative, sr_shift
export sr_diag, sr_cholesky, sr_div, sr_minres, sr_lsq, sr_cg, sr_qlp

@enum SRPreconditionType sr_none=1 sr_shift=2 sr_multiplicative=3
@enum SRAlgorithm sr_diag=1 sr_cholesky=2 sr_pivcholesky=3 sr_div=4 sr_qlp=11 sr_minres=12 sr_lsq=14 sr_cg=15
is_iterative(alg::SRAlgorithm) = Int(alg) >=10

################################################################################
######   Preconditioning algorithm definition (structure holding params)  ######
################################################################################
"""
    SR([use_iterative=true, ϵ=0.001, λ0=100, b=0.95, λmin=1e-4, [precondition_type=sr_shift, algorithm=sr_qlp, precision=1e-4])

Stochastic Reconfiguration preconditioner which corrects the gradient according
to the natural gradient computed as S^-1 ∇C. Using this algorithm will lead to
the computation of the S matrix together with the gradient of the cost function
∇C. To compute the natural gradient S^-1∇C an iterative scheme (Minres-QLP) or
a direct inversion is used.

The linear system x = S^-1 ∇C is by default solved with `minres_qlp` iterative
solver. Alternatively you can use `sr_minres`, `sr_cg` or `sr_lsq` to use respectively
the minres, conjugate gradient and least square solvers from IterativeSolvers.jl.
For small systems you can also solve it by computing the pseudo-inverse (`sr_diag`),
the cholesky factorisation (`sr_cholesky`) the pivoted-cholesky factorisation (`sr_pivcholesky`),
and using the automatic julia solver, usually involving qr decomposition (`sr_div`).
Those non-iterative methods are all from Base.LinearAlgebra.

If `use_iterative=true` the inverse matrix `S^-1` is not computed, and an iterative
MINRES-QLP algorithm is used to compute the product S^-1*F

If `precondition_type=sr_shift` then a diagonal uniform shift is added to S
S --> S+ϵ*identity

If `precondition_type=sr_multiplicative` then a diagonal multiplicative shift is added to S
S --> S + max(λ0*b^n,λmin)*Diagonal(diag(S)) where n is the number of the iteration.
"""
struct SR{T1,T2,T3,TP} <: Algorithm
    sr_diag_shift::T1  # cutoff
    sr_diag_mult::T2
    sr_precision::T3   #cutoff
    precondition_type::SRPreconditionType
    algorithm::SRAlgorithm
    use_fullmat::Bool
    λ0::TP
    b::TP
    λmin::TP
end

SR(T::Type=STD_REAL_PREC; ϵ=0.001, precision=10e-5,
   precondition_type=sr_shift, full_matrix=true, algorithm=sr_cholesky,
   λ0=100.0, b=0.95, λmin=1e-4) = SR(T(ϵ), T(1.0), T(precision),
                                          precondition_type, algorithm, full_matrix,
                                          T(λ0), T(b), T(λmin))

is_iterative(alg::SR) = is_iterative(algorithm(alg))
algorithm(alg::SR) = alg.algorithm
# -------------- Base.show extension for nice printing -------------- #
Base.show(io::IO, mm::MIME"text/plain", sr::SR) = begin
    print(io, "SR{$(typeof(sr.sr_diag_shift)), $(typeof(sr.λ0))}\n")
    print(io, "\tActive preconditioner: $(sr.precondition_type)\n")
    if sr.precondition_type == sr_multiplicative
        print(io, "\t - λ0   : $(sr.λ0)\n")
        print(io, "\t - b    : $(sr.b)\n")
        print(io, "\t - λmin : $(sr.λmin)")
    else
        print(io, "\t - ϵ : $(sr.sr_diag_shift)")
    end
end

function algorithm_cache(alg::SR, prob, net, par_cache)
    if is_iterative(alg)
        _sr_iterative_cache(alg, prob, net, par_cache)
    else
        _sr_direct_cache(alg, prob, net, par_cache)
    end
end
