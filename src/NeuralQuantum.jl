module NeuralQuantum

# Using statements
using Reexport

using QuantumOptics
using LightGraphs

using Zygote: gradient, forward
using Random: AbstractRNG, MersenneTwister, GLOBAL_RNG
using LinearAlgebra, SparseArrays

include("IterativeSolvers/minresqlp.jl")
using .MinresQlp

# Quantum Lattices, used to construct lattice hamiltonians (custom package)
@reexport using QuantumLattices

# Optimisers, that will be split in a separate package at some point
include("Optimisers/Optimisers.jl")
using .Optimisers
import .Optimisers: update, update!
export Optimisers

# Logging
using TensorBoardLogger, ValueHistoriesLogger

# Imports
import Base: length, UInt, eltype, copy, deepcopy, iterate
import Random: rand!
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

abstract type Sampler end

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
include("States/StateChanges.jl")
include("States/NAryState.jl")
include("States/DoubleState.jl")
include("States/PurifiedState.jl")
include("States/DiagonalStateWrapper.jl")
export local_index
include("States/ModifiedState.jl")
export ModifiedState

# Linear Operators
import Base: +
include("Operators/BaseOperators.jl")
include("Operators/OpConnection.jl")
include("Operators/OpConnectionIndex.jl")
include("Operators/KLocalOperator.jl")
include("Operators/KLocalOperatorSum.jl")
include("Operators/GraphConversion.jl")
export OpConnection
export KLocalOperator, KLocalOperatorSum, KLocalOperatorRow, operators
export row_valdiff, row_valdiff_index, col_valdiff, sites, conn_type
export duplicate


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

# Problems
export LdagL_spmat_prob, LdagL_sop_prob, LdagLProblem, LdagLFullProblem, LdagL_L_prob, LdagL_L_Problem, LdagL_Lmat_prob
include("Problems/LdagL_spmat_prob.jl")
include("Problems/LdagL_sop_prob.jl")
include("Problems/ObservablesProblem.jl")
include("Problems/LdagL_L_prob.jl")
include("Problems/LdagL_Lmat_prob.jl")
include("Problems/Operators/LdagL_Lrho_op_prob.jl")
const LdagLFullProblem = LdagL_sop_prob
const LdagLProblem = LdagL_spmat_prob
const LdagL_L_Problem = LdagL_Lmat_prob

# gen state
export state, state_lut
include("generate_state.jl")

#
export HamProblem
include("Problems/Ham_spmat_prob.jl")
const HamProblem = Ham_spmat_prob

# Algorithms
abstract type Algorithm end
abstract type EvaluatedAlgorithm end
abstract type EvaluationSamplingCache end
include("Algorithms/base_algorithms.jl")
export EvaluatedNetwork, evaluation_post_sampling!, precondition!, SamplingCache

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
export cache, init_sampler!, done, samplenext!
export get_sampler, sampler_list, multithread

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
include("utils/loading.jl")

# interface
include("IterativeInterface/BaseIterativeSampler.jl")
include("IterativeInterface/IterativeSampler.jl")
include("IterativeInterface/MTIterativeSampler.jl")
export sample!

end # module
