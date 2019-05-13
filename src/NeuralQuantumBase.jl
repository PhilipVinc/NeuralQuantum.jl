module NeuralQuantumBase

# Using statements
using Reexport

using QuantumOptics
@reexport using QuantumLattices
using LightGraphs
using Optimisers

include("IterativeSolvers/minresqlp.jl")
using .MinresQlp

using Zygote: gradient, forward

using Random: AbstractRNG, MersenneTwister, GLOBAL_RNG
using LinearAlgebra, SparseArrays

# Logging
using TensorBoardLogger, ValueHistoriesLogger


# Imports
import Base: length, UInt, eltype, copy, deepcopy, iterate
import Random: rand!
import Optimisers: update, update!, apply!
import QuantumOptics: basis

# Abstract Types
abstract type NeuralNetwork end

abstract type State end
abstract type FiniteBasisState <: State end

abstract type Problem end
abstract type SteadyStateProblem <: Problem end
abstract type HermitianMatrixProblem <: SteadyStateProblem end
abstract type LRhoSquaredProblem <: SteadyStateProblem end
abstract type OpenTimeEvolutionProblem <: SteadyStateProblem end
abstract type OperatorEstimationProblem <: Problem end

# Type describing the parallel backend used by a solver.
abstract type ParallelType end
struct NotParallel <: ParallelType end
struct ParallelThreaded <: ParallelType end
export NotParallel, ParallelThreaded

# Base elements
include("base_states.jl")
include("base_networks.jl")
include("base_cached_networks.jl")
include("tuple_logic.jl")

# Basic states for uniform systems
include("States/NAryState.jl")
include("States/DoubleState.jl")
include("States/PurifiedState.jl")
include("States/DiagonalStateWrapper.jl")

# Neural Networks
include("Networks/utils.jl")
include("Networks/RBMSplit.jl")
include("Networks/NDM.jl")
include("Networks/NDMComplex.jl")
include("Networks/NDMSymm.jl")

const rRBMSplit = RBMSplit; export rRBMSplit;
const rNDM = NDM; export rNDM;
const rNDMSymm = NDMSymm; export rNDMSymm;

#
include("Networks/ClosedSystems/RBM.jl")

# gen state
export state
include("generate_state.jl")

# Problems
export LdagL_spmat_prob, LdagL_sop_prob, LdagLProblem, LdagLFullProblem, LdagL_L_prob, LdagL_L_Problem, LdagL_Lmat_prob
include("Problems/LdagL_spmat_prob.jl")
include("Problems/LdagL_sop_prob.jl")
include("Problems/ObservablesProblem.jl")
include("Problems/LdagL_L_prob.jl")
include("Problems/LdagL_Lmat_prob.jl")
const LdagLFullProblem = LdagL_sop_prob
const LdagLProblem = LdagL_spmat_prob
const LdagL_L_Problem = LdagL_Lmat_prob

#
export HamProblem
include("Problems/Ham_spmat_prob.jl")
const HamProblem = Ham_spmat_prob

# Algs
abstract type Algorithm end
abstract type EvaluatedAlgorithm end
abstract type EvaluationSamplingCache end
include("Algorithms/base_algorithms.jl")
# SR
include("Algorithms/SR/SR.jl")
include("Algorithms/SR/SampledSRCache.jl")
include("Algorithms/SR/SampledSRCache_L.jl")
include("Algorithms/SR/SR_eval.jl")
# Gradient
include("Algorithms/Gradient/Gradient.jl")
include("Algorithms/Gradient/SampledGradientCache.jl")
include("Algorithms/Gradient/SampledGradientCache_L.jl")
include("Algorithms/Gradient/Gradient_eval.jl")
# Observables
include("Algorithms/Observables/Obs.jl")
include("Algorithms/Observables/Obs_eval.jl")


# Sampling
include("Samplers/base_samplers.jl")
include("Samplers/base_samplers_parallel.jl")
include("Samplers/Exact.jl")
include("Samplers/FullSum.jl")
include("Samplers/MCMCSampler.jl")
include("Samplers/MCMCRules/Metropolis.jl")
include("Samplers/MCMCRules/Nagy.jl")

# other
include("utils/densitymatrix.jl")
include("utils/expectation_values.jl")
include("utils/translational_symm.jl")
include("utils/logging.jl")

# interface
include("IterativeInterface/BaseIterativeSampler.jl")
include("IterativeInterface/IterativeSampler.jl")
include("IterativeInterface/MTIterativeSampler.jl")

end # module
