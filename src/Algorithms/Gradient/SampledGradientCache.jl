################################################################################
######      Cache holding the information generated along a sampling      ######
################################################################################
mutable struct MCMCGradientEvaluationCache{T,T2,TV,TVC,TD, S} <: EvaluationSamplingCache
    Oave::TV#Vector{T}
    Eave::T
    EOave::TVC
    Zave::T2

    # Individual values to compute statistical correlators
    Evalues::Vector{T}

    # Caches to avoid allocating during computation
    ∇lnψ::TD
    σ::S
end

function MCMCGradientEvaluationCache(net::NeuralNetwork, prob)
    TC = Complex{real(out_type(net))}
    der_vec = grad_cache(net).tuple_all_weights

    Oave  = [zero(dvec) for dvec=der_vec]
    EOave = [zeros(TC, size(dvec)) for dvec=der_vec]

    cache = MCMCGradientEvaluationCache(Oave,
                                        zero(TC),
                                        EOave,
                                        zero(real(TC)),
                                        Vector{TC}(),
                                        grad_cache(net),
                                        state(prob, net))
    zero!(cache)
end
SamplingCache(alg::Gradient, prob::HermitianMatrixProblem,   net) = MCMCGradientEvaluationCache(net, prob)
SamplingCache(alg::Gradient, prob::OpenTimeEvolutionProblem, net) = throw(ErrorException("Can't do time evolution with gradient. need SR."))


function zero!(comp_vals::MCMCGradientEvaluationCache)
    comp_vals.Eave   = 0.0
    comp_vals.Zave = 0.0
    resize!(comp_vals.Evalues, 0)

    for (Oave, EOave) = zip(comp_vals.Oave, comp_vals.EOave)
        Oave  .= 0.0
        EOave .= 0.0
    end

    comp_vals
end

# Utility method utilised to accumulate results on a single variable
function add!(acc::MCMCGradientEvaluationCache, o::MCMCGradientEvaluationCache)
    acc.Eave   += o.Eave
    acc.Zave   += o.Zave
    append!(acc.Evalues, o.Evalues)

    for i=1:length(acc.Oave)
       acc.Oave[i]  .+= o.Oave[i]
       acc.EOave[i] .+= o.EOave[i]
   end
   acc
end

#TODO Things that allocate in this function are
# out.LVals ./= 1:length(vals.Evalues)
# F .= real.(EOave .- Eave .* conj(Oave))
# and they clearly should not. It's julia's fault.
##
function evaluation_post_sampling!(out::GradientEvaluation,
                                   vals::MCMCGradientEvaluationCache,
                                   sampler_steps = vals.Zave)
     Eave = vals.Eave   /= sampler_steps

     #TODO Here I am reallocating. Should think about how to fix it.
     out.L = Eave
     # Convergence history
     #out.LVals = cumsum(vals.Evalues)./(1:length(vals.Evalues))
     resize!(out.LVals, length(vals.Evalues))
     cumsum!(out.LVals, vals.Evalues)
     out.LVals ./= 1:length(vals.Evalues)

     for i=1:length(vals.Oave)
         Oave  = vals.Oave[i]  ./= sampler_steps
         EOave = vals.EOave[i] ./= sampler_steps

         F = out.F[i];
         if eltype(F) <: Real
             F .= real.(EOave .- Eave .* conj(Oave))
         else
             F .=       EOave .- Eave .* conj(Oave)
         end
     end
end
