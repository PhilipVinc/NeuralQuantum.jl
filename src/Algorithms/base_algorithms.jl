add!(a::EvaluatedAlgorithm, b::EvaluatedAlgorithm) = error("NOT IMPLEMENTED: add!($(typeof(a)), $(typeof(b)))")

#zero!
#add!

"""
    sample_network_wholespace!(res, prob, net, σ)

Adds to the MonteCarlo sampling cache `res` a sample of problem `prob` taken
from network `net`, with state `σ`.  Any quantity added is multiplied by the
probability `|net(σ)|^2`. Used by `FullSumSampler`.
"""
sample_network_wholespace!(res, prob, net, σ) =
  sample_network!(res, prob, net, σ, true)

"""
    sample_network_wholespace!(res, prob, net, σ)

Adds to the MonteCarlo sampling cache `res` a sample of problem `prob` taken
from network `net`, with state `σ`.
"""
function sample_network! end

"""
	EvaluatedNetwork(algorithm, net)

Constructs the EvaluatedNetwork structure holding the quantities that must
be computed to apply `algorithm` to `net`.
"""
function EvaluatedNetwork end

"""
	SamplingCache(algorithm, problem, net)

Constructs the cache to hold quantities necessary to sample the `problem` with
the given `algorithm`.
"""
function SamplingCache end
