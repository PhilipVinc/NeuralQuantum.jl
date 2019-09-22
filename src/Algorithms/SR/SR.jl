export SR, sr_none, sr_multiplicative, sr_shift

@enum SRPreconditionType sr_none=1 sr_shift=2 sr_multiplicative=3
################################################################################
######   Preconditioning algorithm definition (structure holding params)  ######
################################################################################
"""
    SR([use_iterative=true, ϵ=0.001, λ0=100, b=0.95, λmin=1e-4, precondition_type=sr_shift)

Stochastic Reconfiguration preconditioner which corrects the gradient according
to the natural gradient computed as S^-1 ∇C. Using this algorithm will lead to
the computation of the S matrix together with the gradient of the cost function
∇C. To compute the natural gradient S^-1∇C an iterative scheme (Minres-QLP) or
a direct inversion is used.

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
    use_iterative::Bool
    λ0::TP
    b::TP
    λmin::TP
end

SR(T::Type=STD_REAL_PREC; ϵ=0.001, use_iterative=true, precision=10e-5,
   precondition_type=sr_shift,
   λ0=100.0, b=0.95, λmin=1e-4) = SR(T(ϵ), T(1.0), T(precision),
                                          precondition_type, use_iterative,
                                          T(λ0), T(b), T(λmin))

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

#Base.show(io::IO, sr::SR) = print(io, "(",bs.σ_row,", ", bs.σ_col, ")")


################################################################################
######  Structure holding information computed at the end of a sampling   ######
################################################################################
"""
The SREvaluation structure holds the evaluation of the Network, L, the
vector of generalized forces F acting on it, and the change of basis matrix
S. See S.Sorella et al.
"""
mutable struct SREvaluation{TL,TF,TS} <: EvaluatedAlgorithm
    L::TL
    F::TF
    S::TS

    # Individual values to compute statistical correlators
    LVals::Vector
end

function SREvaluation(net::NeuralNetwork)
    wt = grad_cache(net) # this should be weight_type
    WT = weight_type(net)
    T = out_type(net)


    F = Tuple([zeros(WT,size(w)) for w=wt.tuple_all_weights])
    S = Tuple([zeros(WT,size(w*w')) for w=wt.tuple_all_weights])


    SREvaluation(zero(T),
                 F,
                 S,
                 Vector{T}())
end

EvaluatedNetwork(alg::SR, net) =
    SREvaluation(weights(net))

# Utility method utilised to accumulate results on a single variable
function add!(acc::SREvaluation, o::SREvaluation)
   acc.L  += o.L
   acc.F .+= o.F
   acc.S .+= o.S

   append!(acc.LVals, o.LVals)
end

function precondition!(∇x, params::SR, data::SREvaluation, iter_n)
    ϵ = params.sr_diag_shift

    success = true

    for (Δw, S, F) = zip(∇x, data.S, data.F)
        # new matrix
        if params.precondition_type == sr_none
            Sprecond = S
        elseif params.precondition_type == sr_shift
            Sprecond = S + convert(eltype(S), ϵ)*I
        elseif params.precondition_type == sr_multiplicative
            λ0 = params.λ0; b = params.b; λmin = params.λmin;
            #vv = eigvals(S)
            #println("regol $(max(λ0*b^iter_n, λmin)) --> $(minimum(vv)), a $(maximum(vv))")
            λ = convert(eltype(S),max(λ0*b^iter_n, λmin))
            Sprecond = S + λ*Diagonal(diag(S))

        end

        if !params.use_iterative
            try
                Δw .= pinv(Sprecond)*F
            catch err
                println("Could not invert: $err")
                Δw .= 0.0
                success = false
            end
        else
            x, hist = minresqlp(Sprecond, F, maxiter=size(S,2)*10, log=true, verbose=false, tol=params.sr_precision)
            #x, hist = cg(S.+ ϵ*I, F, maxiter=size(S,2)*10, log=true, verbose=true, tol=10e-10)
            Δw .= x
            add_iters = 1
            while !hist.isconverged
                println("minresqlp not conerged. Additional $(size(S,2)*10) iters for the $add_iters time.")
                x, hist = minresqlp(ΔW, Sprecond, F, maxiter=size(S,2)*10, log=true)
                Δw .= x
                add_iters += 1
                add_iters > 5 && break
            end
            if add_iters > 5 && !hist.isconverged
                success = false
                Δw .= 0.0
            end
        end
    end
    return success
end
