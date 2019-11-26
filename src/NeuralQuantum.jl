module NeuralQuantum

# Using statements
using Reexport
using Requires
using MacroTools: @forward

using CuArrays
const use_cuda = Ref(false)

using QuantumOpticsBase
using LightGraphs
using Zygote
using NNlib
using LinearMaps
using Random: Random, AbstractRNG, MersenneTwister, GLOBAL_RNG, rand!
using LinearAlgebra, SparseArrays, Strided, UnsafeArrays
using Statistics
using Printf

include("IterativeSolvers/minresqlp.jl")
using .MinresQlp

# Quantum Lattices, used to construct lattice hamiltonians (custom package)
@reexport using QuantumLattices

# Optimisers, that will be split in a separate package at some point
include("Optimisers/Optimisers.jl")
using .Optimisers
import .Optimisers: update, update!
export Optimisers

# Abstract Types
abstract type NeuralNetwork end

abstract type AbstractHilbert end

abstract type State end
abstract type FiniteBasisState <: State end

abstract type Sampler end
abstract type AbstractAccumulator end

abstract type AbstractProblem end
include("Problems/base_problems.jl")

# Type describing the parallel backend used by a solver.
abstract type ParallelType end
struct NotParallel <: ParallelType end
struct ParallelThreaded <: ParallelType end
export NotParallel, ParallelThreaded

# Universal defines
const STD_REAL_PREC =  Float32

# Basic states for uniform systems
include("base_states.jl")
include("States/StateChanges.jl")

# Base elements
include("Hilbert/base_basis.jl")
include("base_derivatives.jl")
include("base_networks.jl")
include("base_cached_networks.jl")
include("treelike.jl") # from flux
include("tuple_logic.jl")

# Useful
include("utils/math.jl")
include("utils/stats.jl")

#=
include("States/NAryState.jl")
include("States/DoubleState.jl")
include("States/PurifiedState.jl")
include("States/DiagonalStateWrapper.jl")
include("States/ModifiedState.jl")
export ModifiedState, local_index
=#

include("Hilbert/DiscreteHilbert.jl")
include("Hilbert/HomogeneousHilbert.jl")
include("Hilbert/SuperHilbert.jl")
include("Hilbert/basis_convert.jl")

include("base_batched_networks.jl")

# Linear Operators
import Base: +, *
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

include("Operators/GraphConversion.jl")
export OpConnection
export KLocalOperator, KLocalOperatorTensor, KLocalOperatorSum, KLocalOperatorRow, operators
export row_valdiff, row_valdiff_index, col_valdiff, sites, conn_type
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
include("Networks/ClosedSystems/Chain.jl")
include("Networks/ClosedSystems/SimpleLayers.jl")
include("Networks/ClosedSystems/conv.jl")
include("Networks/ClosedSystems/NQConv.jl")

# Wrappers
include("Networks/NetworkWrappers.jl")

# Problems
include("Problems/SteadyStateLindblad/LRhoKLocalSOpProblem.jl")

include("Problems/SteadyStateLindblad/build_SteadyStateProblem.jl")

# Hamiltonian problems
include("Problems/Hamiltonian/HamiltonianGSEnergyProblem.jl")
include("Problems/Hamiltonian/HamiltonianGSVarianceProblem.jl")
include("Problems/Hamiltonian/build_GroundStateProblem.jl")

# Observables problem
include("Problems/ObservablesProblem.jl")


# gen state
export state
include("generate_state.jl")

# Algorithms
abstract type Algorithm end
abstract type EvaluatedAlgorithm end
abstract type EvaluationSamplingCache end
include("Algorithms/base_algorithms.jl")
export EvaluatedNetwork, evaluation_post_sampling!, precondition!, SamplingCache

# SR
include("Algorithms/SR/SR.jl")
# Gradient
include("Algorithms/Gradient/Gradient.jl")
# Observables
include("Algorithms/Observables/Obs.jl")

include("Algorithms/batched_algorithms.jl")
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
include("Samplers/MCMCRules/Nagy.jl")

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

export sample!, add_observable, compute_observables

include("utils/num_grad.jl")

function __init__()
    @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
        import .CuArrays: CuArrays, @cufunc

    #=    @cufunc NeuralQuantum.ℒ(x) = one(x) + exp(x)

        @cufunc NeuralQuantum.∂logℒ(x) = one(x)/(one(x)+exp(-x))

        @cufunc NeuralQuantum.logℒ(x::Real) = log1p(exp(x))
        @cufunc NeuralQuantum.logℒ(x::Complex) = log(one(x) + exp(x))=#
    end

    @require QuantumOptics="6e0679c1-51ea-5a7c-ac74-d61b76210b0c" begin

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

      # FIXME: this functionality should be conditional at run time by checking `use_cuda`
      #        (or even better, get moved to CuArrays.jl as much as possible)
      if CuArrays.has_cudnn()
        #include(joinpath(@__DIR__, "cuda/cuda.jl"))
      else
        @warn "CuArrays.jl did not find libcudnn. Some functionality will not be available."
      end
    end

    end
end

end # module
