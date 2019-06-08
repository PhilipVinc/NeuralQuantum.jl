#using NeuralQuantum
using Test

@testset "NeuralQuantum" begin

    @testset "Machines" begin
        include("machines.jl")
    end

    @testset "Problem & algs" begin
        include("Problems/problem_algs.jl")
        include("Problems/problem_operators.jl")
    end

    @testset "States" begin
        include("states.jl")
    end

    @testset "Operators" begin
        include("states.jl")
    end

end
