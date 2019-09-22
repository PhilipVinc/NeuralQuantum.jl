using NeuralQuantum: MCMCSREvaluationCache

function test_zero(sc::MCMCSREvaluationCache)
    @testset "MCMCSREvaluationCache zero!" begin
        @test iszero(sc.Oave)
        @test iszero(sc.OOave)
        @test iszero(sc.Eave)
        @test iszero(sc.EOave)
        @test iszero(sc.Zave)

        @test isempty(sc.Evalues)
    end
end

function random_data!(sc::MCMCSREvaluationCache)
    [ rand!(el) for el=sc.Oave]
    [ rand!(el) for el=sc.OOave]
    sc.Eave = rand()
    [ rand!(el) for el=sc.EOave]
    sc.Zave = rand()

    append!(sc.Evalues, rand(ComplexF64, 30))
end

function test_sum!(a::T, b::T, c::T) where T<:MCMCSREvaluationCache
    @test all(a.Oave .+ b.Oave .≈ c.Oave)
    @test all(a.OOave .+ b.OOave .≈ c.OOave)
    @test a.Eave + b.Eave ≈ c.Eave
    @test all(a.EOave .+ b.EOave .≈ c.EOave)
    @test a.Zave + b.Zave ≈ c.Zave

    @test any([vcat(a.Evalues, b.Evalues) ≈ c.Evalues,
               vcat(b.Evalues, a.Evalues) ≈ c.Evalues])
end
