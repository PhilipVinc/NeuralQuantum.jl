export Gradient

################################################################################
######   Preconditioning algorithm definition (structure holding params)  ######
################################################################################
"""
    Gradient()

Algorithm for descending along the steepest gradient with SGD-based optimizers.
"""
struct Gradient <: Algorithm end

################################################################################
######  Structure holding information computed at the end of a sampling   ######
################################################################################
"""
    GradientEvaluation <: EvaluatedAlgorithm

The GradientEvaluation structure holds the evaluation of the Network, L, the
vector of generalized forces F acting on it.
"""
mutable struct GradientEvaluation{TL,TF} <: EvaluatedAlgorithm
    L::TL
    F::TF

    # Individual values to compute statistical correlators
    LVals::Vector
end

function GradientEvaluation(net::NeuralNetwork)
    wt = grad_cache(net)
    WT = weight_type(net)
    T = out_type(net)

    F = Tuple([zeros(WT,size(w)) for w=wt.tuple_all_weights])

    GradientEvaluation(zero(T),
                       F,
                       Vector{T}())
end

EvaluatedNetwork(alg::Gradient, net) =
    GradientEvaluation(weights(net))

# Utility method utilised to accumulate results on a single variable
function add!(acc::GradientEvaluation, o::GradientEvaluation)
   acc.L  += o.L
   acc.F .+= o.F

   append!(acc.LVals, o.LVals)
end

function precondition!(∇x, params::Gradient, data::GradientEvaluation, args...)
    for (Δw, F) = zip(∇x, data.F)
        Δw .= F
    end
    true
end
