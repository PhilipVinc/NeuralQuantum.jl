#using NeuralQuantum
using Test

@testset "NeuralQuantum" begin
    println("Testing machines...")
    @testset "Machines" begin
        include("Machines/test_grad.jl")
        include("Machines/test_lut.jl")
    end

    println("Testing Problems and algorithms...")
    @testset "Problem & algs" begin
        include("Problems/problem_algs.jl")
        include("Problems/problem_operators.jl")
        include("Problems/observables.jl")
    end

    println("Testing States...")
    @testset "States" begin
        include("States/states.jl")
        include("States/modified_states.jl")
    end

    println("Testing Operators...")
    @testset "Operators" begin
        include("operators.jl")
    end

end
