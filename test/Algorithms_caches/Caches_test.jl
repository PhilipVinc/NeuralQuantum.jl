using NeuralQuantum
using LinearAlgebra, SparseArrays
using Test

function test_cache(sc_1)
    @testset "test zero" begin
        zero!(sc_1)
        test_zero(sc_1)
    end

    @testset "test addition" begin
        sc_2 = deepcopy(sc_1)
        random_data!(sc_1)
        random_data!(sc_2)
        sc_3 = deepcopy(sc_1)

        NeuralQuantum.add!(sc_3, sc_2)
        test_sum!(sc_1, sc_2, sc_3)
    end
end

Nsites = 4
lind = quantum_ising_lind(SquareLattice([Nsites],PBC=true), g=1.3, V=2.0, γ=1.0)
net  = cached(NDM(Nsites, 2, 3))

@testset "Test sampling caches" begin
    include("MCMC_SR_cache.jl")
    @testset "Test sampling cache MCMCSREvaluationCache" begin
        prob = SteadyStateProblem(lind, operators=false, variance=false)
        sc   = SamplingCache(SR(), prob, net)
        @test sc isa MCMCSREvaluationCache
        test_cache(sc)
    end

    include("MCMC_SRL_cache.jl")
    @testset "Test sampling cache MCMCSRLEvaluationCache" begin
        prob = SteadyStateProblem(lind, operators=false, variance=true)
        sc   = SamplingCache(SR(), prob, net)
        @test sc isa MCMCSRLEvaluationCache
        test_cache(sc)
    end

    include("MCMC_Grad_cache.jl")
    @testset "Test sampling cache MCMCGradientEvaluationCache" begin
        prob = SteadyStateProblem(lind, operators=false, variance=false)
        sc   = SamplingCache(Gradient(), prob, net)
        @test sc isa MCMCGradientEvaluationCache
        test_cache(sc)
    end

    include("MCMC_GradL_cache.jl")
    @testset "Test sampling cache MCMCGradientLEvaluationCache" begin
        prob = SteadyStateProblem(lind, operators=false, variance=true)
        sc   = SamplingCache(Gradient(), prob, net)
        @test sc isa MCMCGradientLEvaluationCache
        test_cache(sc)
    end
end
