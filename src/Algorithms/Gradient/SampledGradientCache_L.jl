################################################################################
######      Cache holding the information generated along a sampling      ######
################################################################################
mutable struct MCMCGradientLEvaluationCache{T,T2,TV,TVC,TD,S} <: EvaluationSamplingCache
    Oave::TV#Vector{T}
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

function MCMCGradientLEvaluationCache(net::NeuralNetwork, prob)
    TC = Complex{real(out_type(net))}
    der_vec = grad_cache(net).tuple_all_weights

    Oave   = [zero(dvec) for dvec=der_vec]
    EOave  = [zeros(TC, size(dvec)) for dvec=der_vec]
    LLOave = [zeros(TC, size(dvec)) for dvec=der_vec]
    LLO_i  = [zeros(TC, size(dvec)) for dvec=der_vec]

    cache = MCMCGradientLEvaluationCache(Oave,
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

SamplingCache(alg::Gradient, prob::LRhoSquaredProblem, net) = MCMCGradientLEvaluationCache(net, prob)

function zero!(comp_vals::MCMCGradientLEvaluationCache)
    comp_vals.Eave   = 0.0
    comp_vals.Zave = 0.0
    resize!(comp_vals.Evalues, 0)

    for i=1:length(comp_vals.Oave)
        comp_vals.Oave[i]   .= 0.0
        comp_vals.EOave[i]  .= 0.0
        comp_vals.LLOave[i] .= 0.0
    end

    comp_vals
end

# Utility method utilised to accumulate results on a single variable
function add!(acc::MCMCGradientLEvaluationCache, o::MCMCGradientLEvaluationCache)
    acc.Eave   += o.Eave
    acc.Zave   += o.Zave
    append!(acc.Evalues, o.Evalues)

    for i=1:length(acc.Oave)
       acc.Oave[i]   .+= o.Oave[i]
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
function evaluation_post_sampling!(out::GradientEvaluation,
                                   vals::MCMCGradientLEvaluationCache,
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
         EOave  = vals.EOave[i]  ./= sampler_steps
         LLOave = vals.LLOave[i] ./= sampler_steps

         F = out.F[i];
         if eltype(F) <: Real
             F .= real.(LLOave .- Eave .* conj(Oave))
         else
             F .=       LLOave .- Eave .* conj(Oave)
         end
     end
end

#=
function sample_network!(res::MCMCGradientLEvaluationCache, prob::LRhoSquaredProblem,
                         net, σ, wholespace=false)
  CLO_i = res.LLO_i

  lnψ, ∇lnψ = logψ_and_∇logψ!(res.∇lnψ, net, σ)
  C_loc = compute_Cloc!(CLO_i, res.∇lnψ2, prob, net, σ)

  prob = wholespace ? exp(2*real(lnψ)) : 1.0
  E = abs(C_loc)^2

  res.Zave += prob
  res.Eave += prob * E
  push!(res.Evalues, prob*E)

  for (i, _∇lnψ)= enumerate(∇lnψ.tuple_all_weights)
    res.Oave[i]  .+= prob .* _∇lnψ
    res.EOave[i] .+= prob .* conj.(_∇lnψ) .* E
    res.LLOave[i].+= prob .* conj(C_loc) .* CLO_i[i]
  end
  return res
end
=#
