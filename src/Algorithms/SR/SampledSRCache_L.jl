################################################################################
######      Cache holding the information generated along a sampling      ######
################################################################################
mutable struct MCMCSRLEvaluationCache{T,T2,TV,TVC,TM,TD,S} <: EvaluationSamplingCache
    Oave::TV#Vector{T}
    OOave::TM#Matrix{T}
    Eave::T
    EOave::TVC
    LLOave::TVC
    Zave::T2

    # Individual values to compute statistical correlators
    Evalues::Vector{T}

    # Caches to avoid allocating during computation
    LLO_i::TVC
    ∇lnψ::TD
    ∇lnψ2::TD
    σ::S
end

function MCMCSRLEvaluationCache(net::NeuralNetwork, prob)
    TC = Complex{real(out_type(net))}
    der_vec = grad_cache(net).tuple_all_weights

    Oave   = [zero(dvec) for dvec=der_vec]
    OOave  = [zeros(eltype(dvec), length(dvec), length(dvec)) for dvec=der_vec]
    EOave  = [zeros(TC, size(dvec)) for dvec=der_vec]
    LLOave = [zeros(TC, size(dvec)) for dvec=der_vec]
    LLO_i  = [zeros(TC, size(dvec)) for dvec=der_vec]

    cache = MCMCSRLEvaluationCache(Oave,
                                  OOave,
                                  zero(TC),
                                  EOave,
                                  LLOave,
                                  zero(real(TC)),
                                  Vector{TC}(),
                                  LLO_i,
                                  grad_cache(net),
                                  grad_cache(net),
                                  state(prob, net))
    zero!(cache)
end

SamplingCache(alg::SR, prob::LRhoSquaredProblem, net) = MCMCSRLEvaluationCache(net, prob)

function zero!(comp_vals::MCMCSRLEvaluationCache)
    comp_vals.Eave   = 0.0
    comp_vals.Zave = 0.0
    resize!(comp_vals.Evalues, 0)

    for i=1:length(comp_vals.Oave)
        comp_vals.Oave[i]   .= 0.0
        comp_vals.OOave[i]  .= 0.0
        comp_vals.EOave[i]  .= 0.0
        comp_vals.LLOave[i] .= 0.0
    end

    comp_vals
end

# Utility method utilised to accumulate results on a single variable
function add!(acc::MCMCSRLEvaluationCache, o::MCMCSRLEvaluationCache)
    acc.Eave   += o.Eave
    acc.Zave   += o.Zave
    append!(acc.Evalues, o.Evalues)

    for i=1:length(acc.Oave)
       acc.Oave[i]   .+= o.Oave[i]
       acc.OOave[i]  .+= o.OOave[i]
       acc.EOave[i]  .+= o.EOave[i]
       acc.LLOave[i] .+= o.LLOave[i]
   end
   acc
end

#TODO Things that allocate in this function are
# out.LVals ./= 1:length(vals.Evalues)
# S .= real.(OOave .- (conj(Oave) .* transpose(Oave)))
# F .= real.(EOave .- Eave .* conj(Oave))
# and they clearly should not. It's julia's fault.
##
function evaluation_post_sampling!(out::SREvaluation,
                                   vals::MCMCSRLEvaluationCache,
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
         Oave   = vals.Oave[i]   ./= sampler_steps
         OOave  = vals.OOave[i]  ./= sampler_steps
         EOave  = vals.EOave[i]  ./= sampler_steps
         LLOave = vals.LLOave[i] ./= sampler_steps

         S = out.S[i]; F = out.F[i];
         if eltype(S) <: Real
             S .= real.(OOave  .- (conj(Oave) .* transpose(Oave)))
             F .= real.(LLOave .- Eave .* conj(Oave))
         else
             S .=       OOave  .- (conj(Oave) .* transpose(Oave))
             F .=       LLOave .- Eave .* conj(Oave)
         end
     end
end
