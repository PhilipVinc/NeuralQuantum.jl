

################################################################################
######  Structure holding information computed at the end of a sampling   ######
################################################################################
mutable struct SampledObservables{T} <: EvaluatedAlgorithm
    ObsNames::Vector{Symbol}
    ObsAve::Vector{T}
    ObsVals::Vector{Vector{T}}
end

function SampledObservables(oprob, net::NeuralNetwork)
    names = deepcopy(oprob.Names)
    type  = out_type(net)
    for el=oprob.ObservablesTransposed
        type = promote_type(type, eltype(el))
    end
    ObsAve  = zeros(type, length(oprob.ObservablesTransposed))
    ObsVals = [Vector{type}() for i=1:length(oprob.ObservablesTransposed)]

    return SampledObservables(names, ObsAve, ObsVals)
end

EvaluatedNetwork(alg::ObservablesProblem, net) =
    SampledObservables(alg, net)

function add!(acc::SampledObservables, o::SampledObservables)
    for i=1:length(acc.ObsNames)
        @assert acc.ObsNames[i] == o.ObsNames[i]
        acc.ObsAve[i] = acc.ObsAve[i] + o.ObsAve[i]
    end
    acc.ObsAve[i] ./= length(acc.ObsNames)
    acc
end

################################################################################
######      Cache holding the information generated along a sampling      ######
################################################################################
mutable struct MCMCObsEvaluationCache{T,T2,S} <: EvaluationSamplingCache
    ObsNames::Vector{Symbol}
    ObsAve::T
    ObsVals::T2
    Zave::Real

    # caches
    σ::S
end

function MCMCObsEvaluationCache(net::NeuralNetwork, obs_prob::ObservablesProblem)
    n_obs    = length(obs_prob.ObservablesTransposed)
    obs_ave  = Vector{ComplexF64}(undef, n_obs)
    obs_vals = [Vector{ComplexF64}() for i=1:n_obs]

    σ        = state(obs_prob, net)

    cache=MCMCObsEvaluationCache(obs_prob.Names, obs_ave, obs_vals, 0.0, σ)
    zero!(cache)
    cache
end
SamplingCache(alg::ObservablesProblem, prob::ObservablesProblem, net) = MCMCObsEvaluationCache(net, prob)
SamplingCache(prob::ObservablesProblem, net) = MCMCObsEvaluationCache(net, prob)

function zero!(comp_vals::MCMCObsEvaluationCache)
    comp_vals.ObsAve   .= 0.0
    comp_vals.Zave      = 0.0
    resize!.(comp_vals.ObsVals, 0)
end

function add!(acc::MCMCObsEvaluationCache, o::MCMCObsEvaluationCache)
   acc.ObsAve   .+= o.ObsAve
   acc.Zave      += o.Zave
   append!.(acc.ObsVals, o.ObsVals)
end

function evaluation_post_sampling(vals::MCMCObsEvaluationCache)
    obs_ave  = vals.ObsAve./vals.Zave
    obs_vals = [cumsum(ob_t)./(1:length(ob_t)) for
                                    ob_t=vals.ObsVals]

    SampledObservables(vals.ObsNames, obs_ave, obs_vals)
end

function evaluation_post_sampling!(out::SampledObservables,
                                   vals::MCMCObsEvaluationCache,
                                   sampler_steps = vals.Zave)
    out.ObsAve .= vals.ObsAve./vals.Zave

    for (obv_t, ob_t) = zip(out.ObsVals, vals.ObsVals)
        resize!(obv_t, length(ob_t))
        cumsum!(obv_t, ob_t)
        obv_t ./= 1:length(ob_t)
    end
    return out
end
