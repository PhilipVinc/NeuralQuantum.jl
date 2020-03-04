using NeuralQuantum
using Test

N = 4
@testset "Hilbert spaces " begin
    hilb = HomogeneousSpin(N)
    s = state(hilb)
    vals = Int[]
    for i=1:spacedimension(hilb)
        set!(s, hilb, i)
        push!(vals, toint(s, hilb))
    end

    @test all(vals .== 1:spacedimension(hilb))

    shilb = SuperOpSpace(hilb)
    @test physical(shilb) === hilb

    s = state(shilb)
    vals = Int[]
    for i=1:spacedimension(shilb)
        set!(s, shilb, i)
        push!(vals, toint(s, shilb))
    end
    @test all(vals .== 1:spacedimension(shilb))
end

function test_hilb_iterator(hilb)
    it = states(hilb)

    all_states = collect(it)

    @test length(all_states) == spacedimension(hilb)
    @test length(it) == spacedimension(hilb)
    @test eltype(it) == typeof(state(hilb))

    vals = zeros(length(all_states))
    for (i,v) = enumerate(all_states)
        vals[i] = index(hilb, v)
    end

    @test all(vals .== 1:spacedimension(hilb))
end


@testset "Hilbert space iterator" begin
    test_hilb_iterator(HomogeneousSpin(N))
    test_hilb_iterator(HomogeneousFock(N,2))
    test_hilb_iterator(SuperOpSpace(HomogeneousSpin(N)))
end
