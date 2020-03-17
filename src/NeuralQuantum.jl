module NeuralQuantum

# Packages required to construct the module
using Reexport: @reexport
using Requires: @require
using MacroTools: @forward

# Packages required to work-around performance issues
using SpecializeVarargs
using UnsafeArrays

# Packages to support (mostly) GPUs
using GPUArrays
using Adapt: adapt

# Packages for Distributed/Parallelization
using MPI
include("External/TPI/TPI.jl")
using .TPI

# Math stuff
using LinearMaps: LinearMap
using Strided # for gradient of batches of networks
using NNlib
using Zygote: Zygote, Params, @adjoint, pullback

# Standard library
using Random: Random, AbstractRNG, MersenneTwister, GLOBAL_RNG, rand!
using Random: shuffle!, randn!
using LinearAlgebra
using SparseArrays
using Statistics
using Printf

@reexport using QuantumOpticsBase

# Support for Colored Graphs and Lattices
@reexport using LightGraphs
include("External/ColoredGraphs/ColoredGraphs.jl")
using .ColoredGraphs
export HyperCube, translational_symm_table

# Iterative Solvers and custom minres solver
using IterativeSolvers: minres, lsqr, cg
include("External/IterativeSolvers/minresqlp.jl")
using .MinresQlp

# Optimisers, that will be split in a separate package at some point
include("Optimisers/Optimisers.jl")
using .Optimisers
import .Optimisers: update, update!
export Optimisers

# Exports

# Standard Precision used
const STD_REAL_PREC = Float32

# Abstract Types
abstract type NeuralNetwork end
abstract type Sampler end
abstract type AbstractAccumulator end

# Type describing the parallel backend used by a solver.
abstract type ParallelType end
struct NotParallel <: ParallelType end
struct ParallelThreaded <: ParallelType end
struct ParallelMPI <: ParallelType end
export NotParallel, ParallelThreaded

# Prallelization
include("Parallel/base_parallel.jl")
include("Parallel/not_parallel.jl")

# Various utility functions
include("utils/math.jl")
include("utils/views.jl")
include("utils/stats.jl")
include("utils/rng.jl")

# Basic states
include("States/States.jl")
include("States/Iterators.jl")
include("States/StateChanges.jl")
include("States/ApplyStateChanges.jl")

# Hilbert spaces
abstract type AbstractHilbert end
include("Hilbert/base_basis.jl")
include("Hilbert/iterator.jl")
include("Hilbert/DiscreteHilbert.jl")
include("Hilbert/HomogeneousFock.jl")
include("Hilbert/HomogeneousSpin.jl")
include("Hilbert/DoubledHilbert.jl")
include("Hilbert/SuperHilbert.jl")
include("Hilbert/basis_convert.jl")

# Nets
include("base_networks.jl")
include("base_cached_networks.jl")
include("functor.jl") # from flux
include("structuring.jl")
include("tuple_logic.jl")

#AD
include("AD/base_derivatives.jl")
include("AD/RealDerivatives.jl")
include("base_batched_networks.jl")

# Linear Operators
include("Operators/BaseOperators.jl")
# Connections
include("Operators/OpConnections/OpConnection.jl")
include("Operators/OpConnections/OpConnectionIdentity.jl")
include("Operators/OpConnections/OpConnectionTensor.jl")
include("Operators/OpConnections/SuperOpConnection.jl")

include("Operators/OpConnectionIndex.jl")

include("Operators/Operators/KLocalOperator.jl")
include("Operators/Operators/KLocalIdentity.jl")
include("Operators/Operators/KLocalOperatorSum.jl")
include("Operators/Operators/KLocalOperatorTensor.jl")
include("Operators/Operators/KLocalZero.jl")
include("Operators/Operators/KLocalLiouvillian.jl")

include("Operators/SimpleOperators.jl")
include("Operators/OpConversion.jl")

export OpConnection
export KLocalOperator, KLocalOperatorTensor, KLocalOperatorRow, operators
export row_valdiff, col_valdiff, sites, conn_type


# Neural Networks
include("Networks/utils.jl")
include("Networks/activation.jl")

# Mixed Density Matrices
include("Networks/MixedDensityMatrix/NDM.jl")
include("Networks/MixedDensityMatrix/NDMBatched.jl")
include("Networks/MixedDensityMatrix/NDMComplex.jl")
include("Networks/MixedDensityMatrix/NDMSymm.jl")
include("Networks/MixedDensityMatrix/NDMSymmBatched.jl")
include("Networks/MixedDensityMatrix/RBMSplit.jl")
include("Networks/MixedDensityMatrix/RBMSplitBatched.jl")

# Closed Systems
include("Networks/ClosedSystems/RBM.jl")
include("Networks/ClosedSystems/RBMBatched.jl")

# FFNN
include("Networks/Chains/Chain.jl")
include("Networks/Chains/conv.jl")
include("Networks/Chains/NQConv.jl")

include("Networks/Chains/ChainBatched.jl")

# Layers
include("Networks/Chains/Layers/Dense.jl")
include("Networks/Chains/Layers/WeightedSum.jl")
include("Networks/Chains/Layers/sum.jl")
include("Networks/Chains/Layers/DenseSplit.jl")

include("Networks/Chains/Layers/DenseBatched.jl")
include("Networks/Chains/Layers/WeightedSumBatched.jl")
include("Networks/Chains/Layers/sumBatched.jl")
include("Networks/Chains/Layers/DenseSplitBatched.jl")

include("Networks/Chains/Layers/PositiveDefR.jl")
include("Networks/Chains/Layers/PositiveDefRBatched.jl")

# Wrappers
include("Networks/ClosedSystems/PureChainWrapper.jl")
include("Networks/MixedDensityMatrix/MixedChainWrapper.jl")

# gen state
export state
include("generate_state.jl")

# Algorithms
abstract type Algorithm end
abstract type AlgorithmCache end
export precondition!

# SR
include("Algorithms/SR/SR.jl")
# Gradient
include("Algorithms/Gradient/Gradient.jl")

include("Algorithms/batched_algorithms.jl")
include("Algorithms/SR/SR_notfull.jl")
include("Algorithms/SR/SRDirect.jl")
include("Algorithms/SR/SRIterative.jl")
include("Algorithms/Gradient/Gradient_batched.jl")

# Sampling
include("Samplers/base_samplers.jl")
export cache, init_sampler!, done, samplenext!
export get_sampler, sampler_list, multithread

include("Samplers/Exact.jl")
include("Samplers/Metropolis.jl")
include("Samplers/MCMCRules/LocalRule.jl")
include("Samplers/MCMCRules/ExchangeRule.jl")
include("Samplers/MCMCRules/OperatorRule.jl")
include("Samplers/MCMCRules/Nagy.jl")

# other
include("utils/densitymatrix.jl")
#include("utils/expectation_values.jl")
include("utils/translational_symm.jl")
#include("utils/logging.jl")
include("utils/loading.jl")
# interface
include("IterativeInterface/Samplers/BaseIterativeSampler.jl")

include("IterativeInterface/Accumulators/base_accumulators.jl")
include("IterativeInterface/Accumulators/AccumulatorLogPsi.jl")
include("IterativeInterface/Accumulators/AccumulatorLogGradPsi.jl")
include("IterativeInterface/Accumulators/AccumulatorObsScalar.jl")
include("IterativeInterface/Accumulators/AccumulatorObsGrad.jl")

include("IterativeInterface/Samplers/SimpleIterativeSampler.jl")
include("IterativeInterface/Samplers/CostFun/BatchedGradSampler.jl")
include("IterativeInterface/Samplers/CostFun/BatchedValSampler.jl")
include("IterativeInterface/Samplers/Obs/BatchedObsDMSampler.jl")
include("IterativeInterface/Samplers/Obs/BatchedObsKetSampler.jl")
include("IterativeInterface/build_Batched.jl")

export sample!, add_observable!, compute_observables

include("utils/num_grad.jl")

# Parallelization types
include("Parallel/MPI/mpi.jl")
include("Parallel/Threads/threads_base.jl")
include("Parallel/Threads/threads_wrappers.jl")

# gpu stuff
include("GPU/Base/GPUstates.jl")
include("GPU/Base/JLstates.jl")
include("GPU/IterativeInterface/AccumulatorLogPsi.jl")
include("GPU/IterativeInterface/AccumulatorObsScalar.jl")
include("GPU/Samplers/LocalRuleGPU.jl")

include("GPU/gpu.jl")

# Extra
include("Extra/RecipesBase.jl")
include("Extra/ValueHistories.jl")

function __init__()
  @require TensorBoardLogger="899adc3e-224a-11e9-021f-63837185c80f" begin
    include("Extra/Optionals/TensorBoardLogger.jl")
  end

  @require Measurements="eff96d63-e80a-5855-80a2-b1b0885c5ab7" begin
    include("Extra/Optionals/Measurements.jl")
  end
end

end # module
