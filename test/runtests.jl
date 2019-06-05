#using NeuralQuantum
using Test

@testset "NeuralQuantum" begin

    @testset "Machines" begin
        include("machines.jl")
    end

    @testset "Problem & algs" begin
        include("problem_algs.jl")
    end

    @testset "States" begin
        include("states.jl")
    end
end
