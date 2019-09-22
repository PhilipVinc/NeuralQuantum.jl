module NeuralQuantum

# Using statements
using Reexport

using QuantumOptics
using LightGraphs

using Zygote: gradient, forward
using Random: AbstractRNG, MersenneTwister, GLOBAL_RNG
using LinearAlgebra, SparseArrays, Strided
using NNlib

include("IterativeSolvers/minresqlp.jl")
using .MinresQlp

# Quantum Lattices, used to construct lattice hamiltonians (custom package)
@reexport using QuantumLattices

# Optimisers, that will be split in a separate package at some point
include("Optimisers/Optimisers.jl")
using .Optimisers
import .Optimisers: update, update!
export Optimisers

# Imports
import Base: length, UInt, eltype, copy, deepcopy, iterate
import Random: rand!
import QuantumOptics: basis

# Abstract Types
abstract type NeuralNetwork end

abstract type State end
abstract type FiniteBasisState <: State end

abstract type AbstractProblem end
abstract type AbstractSteadyStateProblem <: AbstractProblem end
abstract type HermitianMatrixProblem <: AbstractSteadyStateProblem end
abstract type LRhoSquaredProblem <: AbstractSteadyStateProblem end
abstract type OpenTimeEvolutionProblem <: AbstractSteadyStateProblem end
abstract type OperatorEstimationProblem <: AbstractProblem end

abstract type Sampler end

# Type describing the parallel backend used by a solver.
abstract type ParallelType end
struct NotParallel <: ParallelType end
struct ParallelThreaded <: ParallelType end
export NotParallel, ParallelThreaded

# Universal defines
const STD_REAL_PREC =  Float32

# Base elements
include("base_states.jl")
include("base_derivatives.jl")
include("base_networks.jl")
include("base_cached_networks.jl")
include("base_lookup.jl")
include("base_batched_networks.jl")
include("treelike.jl") # from flux
include("tuple_logic.jl")

# Useful
include("utils/math.jl")

# Basic states for uniform systems
include("States/StateChanges.jl")
include("States/NAryState.jl")
include("States/DoubleState.jl")
include("States/PurifiedState.jl")
include("States/DiagonalStateWrapper.jl")
export local_index
include("States/ModifiedState.jl")
include("States/LUState.jl")
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

# LT
include("Networks/RBMSplitLT.jl")
include("Networks/NDMLT.jl")

# Batch
include("Networks/RBMSplitBatched.jl")

#
include("Networks/ClosedSystems/RBM.jl")

# Problems
export LdagLSparseOpProblem, LRhoSparseSuperopProblem, LdagLProblem, LdagLFullProblem, LdagLSparseSuperopProblem, LdagLSparseSuperopProblemlem, LRhoSparseOpProblem
include("Problems/SteadyStateLindblad/LdagLSparseOpProblem.jl")
include("Problems/SteadyStateLindblad/LdagLSparseSuperopProblem.jl")
include("Problems/SteadyStateLindblad/LRhoKLocalOpProblem.jl")
include("Problems/SteadyStateLindblad/LRhoSparseOpProblem.jl")
include("Problems/SteadyStateLindblad/LRhoSparseSuperopProblem.jl")
const LdagLFullProblem = LRhoSparseSuperopProblem
const LdagLProblem = LdagLSparseOpProblem
const LdagLSparseSuperopProblemlem = LRhoSparseOpProblem

include("Problems/SteadyStateLindblad/build_SteadyStateProblem.jl")

# Hamiltonian problems
include("Problems/Hamiltonian/HamiltonianGSEnergyProblem.jl")
include("Problems/Hamiltonian/build_GroundStateProblem.jl")

# Observables problem
include("Problems/ObservablesProblem.jl")


# gen state
export state, state_lut
include("generate_state.jl")

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
include("Algorithms/Observables/Obs_ket_eval.jl")


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

include("base_diffeval.jl")

# other
include("utils/densitymatrix.jl")
include("utils/expectation_values.jl")
include("utils/translational_symm.jl")
#include("utils/logging.jl")
include("utils/loading.jl")

# interface
include("IterativeInterface/BaseIterativeSampler.jl")
include("IterativeInterface/IterativeSampler.jl")
include("IterativeInterface/MTIterativeSampler.jl")
export sample!

end # module
