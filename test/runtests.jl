#using NeuralQuantum
using Test

@testset "NeuralQuantum" begin

    @testset "Machines" begin
        include("Machines/test_grad.jl")
        include("Machines/test_lut.jl")
    end

    @testset "Problem & algs" begin
        include("Problems/problem_algs.jl")
        include("Problems/problem_operators.jl")
    end

    @testset "States" begin
        include("States/states.jl")
        include("States/modified_states.jl")
    end

    @testset "Operators" begin
        include("operators.jl")
    end

end
