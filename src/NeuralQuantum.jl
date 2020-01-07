module NeuralQuantum

using Reexport
using Requires
using MacroTools: @forward

using GPUArrays
using CuArrays
const use_cuda = Ref(false)

# Standard Precision used
const STD_REAL_PREC = Float32

using Random: Random, AbstractRNG, MersenneTwister, GLOBAL_RNG, rand!, randn!
using LinearAlgebra
using SparseArrays
using LinearMaps
using Strided
using UnsafeArrays
using Statistics

using Zygote
using NNlib

using Printf

@reexport using QuantumOpticsBase

# Support for Colored Graphs
using LightGraphs
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

# Abstract Types
abstract type AbstractHilbert end
abstract type NeuralNetwork end
abstract type Sampler end
abstract type AbstractAccumulator end

# Type describing the parallel backend used by a solver.
abstract type ParallelType end
struct NotParallel <: ParallelType end
struct ParallelThreaded <: ParallelType end
export NotParallel, ParallelThreaded

# Various utility functions
include("utils/math.jl")
include("utils/stats.jl")
include("utils/rng.jl")

# Basic states for uniform systems
include("base_states.jl")
include("States/StateChanges.jl")

# Base elements
include("Hilbert/base_basis.jl")

# Nets
include("base_networks.jl")
include("base_cached_networks.jl")
include("treelike.jl") # from flux
include("tuple_logic.jl")

#AD
include("AD/base_derivatives.jl")
include("AD/RealDerivatives.jl")

include("Hilbert/DiscreteHilbert.jl")
include("Hilbert/HomogeneousFock.jl")
include("Hilbert/HomogeneousSpin.jl")
include("Hilbert/SuperHilbert.jl")
include("Hilbert/basis_convert.jl")

include("base_batched_networks.jl")

# Linear Operators
include("Operators/BaseOperators.jl")
include("Operators/OpConnection.jl")
include("Operators/OpConnectionIndex.jl")
include("Operators/KLocalOperator.jl")
include("Operators/KLocalOperatorSum.jl")
include("Operators/KLocalOperatorTensor.jl")
include("Operators/KLocalZero.jl")
include("Operators/KLocalLiouvillian.jl")

include("Operators/SimpleOperators.jl")
include("Operators/OpConversion.jl")

export OpConnection
export KLocalOperator, KLocalOperatorTensor, KLocalOperatorRow, operators
export row_valdiff, col_valdiff, sites, conn_type
export duplicate


# Neural Networks
include("Networks/utils.jl")

# Mixed Density Matrices
include("Networks/MixedDensityMatrix/NDM.jl")
include("Networks/MixedDensityMatrix/NDMBatched.jl")
include("Networks/MixedDensityMatrix/NDMComplex.jl")
include("Networks/MixedDensityMatrix/NDMSymm.jl")
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
include("Algorithms/SR/SR_batched.jl")
include("Algorithms/Gradient/Gradient_batched.jl")

# Sampling
include("Samplers/base_samplers.jl")
include("Samplers/base_samplers_parallel.jl")
export cache, init_sampler!, done, samplenext!
export get_sampler, sampler_list, multithread

include("Samplers/Exact.jl")
include("Samplers/Metropolis.jl")
include("Samplers/MCMCRules/LocalRule.jl")
include("Samplers/MCMCRules/ExchangeRule.jl")
include("Samplers/MCMCRules/Nagy.jl")

# other
include("utils/densitymatrix.jl")
#include("utils/expectation_values.jl")
include("utils/translational_symm.jl")
#include("utils/logging.jl")
include("utils/loading.jl")

# interface
include("IterativeInterface/BaseIterativeSampler.jl")
#include("IterativeInterface/IterativeSampler.jl")
#include("IterativeInterface/MTIterativeSampler.jl")

include("IterativeInterface/Batched/base_accumulators.jl")
include("IterativeInterface/Batched/AccumulatorLogPsi.jl")
include("IterativeInterface/Batched/AccumulatorLogGradPsi.jl")
include("IterativeInterface/Batched/AccumulatorObsScalar.jl")
include("IterativeInterface/Batched/AccumulatorObsGrad.jl")

include("IterativeInterface/BatchedGradSampler.jl")
include("IterativeInterface/BatchedValSampler.jl")
include("IterativeInterface/BatchedObsDMSampler.jl")
include("IterativeInterface/BatchedObsKetSampler.jl")
include("IterativeInterface/build_Batched.jl")

export sample!, add_observable!, compute_observables

include("utils/num_grad.jl")

# gpu stuff
include("GPU/gpustates.jl")
include("GPU/LocalRuleGPU.jl")
include("GPU/AccumulatorLogPsi.jl")
include("GPU/AccumulatorObsScalar.jl")

include("GPU/gpuarrays.jl")

function __init__()

  # cuda stuff
  precompiling = ccall(:jl_generating_output, Cint, ()) != 0

  # we don't want to include the CUDA module when precompiling,
  # or we could end up replacing it at run time (triggering a warning)
  precompiling && return

  if !CuArrays.functional()
    # nothing to do here, and either CuArrays or one of its dependencies will have warned
  else
    use_cuda[] = true
    include(joinpath(@__DIR__, "GPU/cuda.jl"))
    include(joinpath(@__DIR__, "GPU/upstream.jl"))

    # FIXME: this functionality should be conditional at run time by checking `use_cuda`
    #        (or even better, get moved to CuArrays.jl as much as possible)
    if CuArrays.has_cudnn()
      #include(joinpath(@__DIR__, "cuda/cuda.jl"))
    else
      @warn "CuArrays.jl did not find libcudnn. Some functionality will not be available."
    end
  end
end

end # module
