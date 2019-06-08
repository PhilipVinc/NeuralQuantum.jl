using NeuralQuantum
using Test

@testset "NAryState" begin
    import NeuralQuantum: NAryState, flipat!, setat!

    s = NAryState(3, 4)
    @test eltype(config(s)) == Float32
    s = NAryState(Float64, 2, 3)
    @test eltype(config(s)) == Float64
    @test nsites(s) == 3
    @test local_dimension(s) == 2
    @test spacedimension(s) == 2^3
    @test toint(s) == 0
    set!(s, 12)
    @test toint(s) == 12
    set_index!(s, 13)
    @test index(s) == 13
    set!(s, spacedimension(s))
    @test all(config(s) .== 0.0)
    @test config(s) isa AbstractArray

    s = NAryState(Float64, 4, 3)
    @test nsites(s) == 3
    @test local_dimension(s) == 4
    @test spacedimension(s) == 4^3
    for i=1:spacedimension(s)
        set_index!(s, i)
        @test index(s) == i
    end
    set_index!(s, 25)
    @test index(s) == 25
    set!(s, spacedimension(s))
    @test all(config(s) .== 0.0)
    @test config(s) isa AbstractArray

    rand!(s)
    for i=1:length(config(s))
        v = config(s)[i]
        old, new = flipat!(s, i)
        @test old == v
        @test new != v
        setat!(s, i, old)
        @test config(s)[i] == v
    end
end

@test "localindex" begin
    import NeuralQuantum: NAryState, flipat!, setat!

    v = NAryState(2,4)
end
