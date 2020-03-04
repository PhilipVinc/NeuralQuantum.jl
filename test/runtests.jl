#using NeuralQuantum
using Test

@testset "NeuralQuantum" begin

    println("Testing Operators...")
    @testset "Operators" begin
        include("Operators/operators.jl")
        include("Operators/ising.jl")
    end

    println("Testing machines...")
    @testset "Machines" begin
        include("Machines/test_cached.jl")
        include("Machines/test_grad.jl")
        include("Machines/test_batched.jl")
        include("Machines/test_ndmcomplex.jl")
    end

    println("Testing samplers...")
    @testset "Samplers" begin
        include("Samplers/test_samplers.jl")
    end

    #=
    println("Testing algorithm caches...")
    @testset "Algorithm" begin
        include("Algorithms_caches/Caches_test.jl")
    end

    println("Testing Problems and algorithms...")
    @testset "Problem & algs" begin
        include("Problems/problem_algs.jl")
        include("Problems/problem_operators.jl")
        include("Problems/observables.jl")
        include("Problems/hamiltonian.jl")
    end
    =#
end
