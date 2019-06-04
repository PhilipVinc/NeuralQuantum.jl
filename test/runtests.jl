#using NeuralQuantum
using Test

@testset "NeuralQuantum" begin

    @testset "Machines" begin
    	@test true == true
        include("machines.jl")
    end

    @testset "Problem & algs" begin
        include("problem_algs.jl")
    end
end
