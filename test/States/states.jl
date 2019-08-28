using NeuralQuantum
using Test
using NeuralQuantum: flipat!

function test_state_properties(s, _ldim, _n_sites, _type, _ctype)
    @test local_dimension(s) == _ldim
    @test nsites(s) == _n_sites
    @test spacedimension(s) == local_dimension(s)^nsites(s)
    @test eltype(s) == _type
    @test typeof(config(s)) <: _ctype
end

function test_state_dyn_single(s)
    l = local_dimension(s)
    n = nsites(s)
    dim = spacedimension(s)

    configurations = [];
    indices = [];
    for i=1:dim
        set_index!(s, i)
        push!(indices, index(s))
        push!(configurations, config(i))
    end

    @test indices == 1:dim
    @test length(unique(configurations)) == dim

    for rep=1:30
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
end

function test_state_dyn_double(s)
    l = local_dimension(s)
    n = nsites(s)
    dim = spacedimension(s)

    configurations = [];
    indices = [];
    for i=1:dim
        set_index!(s, i)
        push!(indices, index(s))
        push!(configurations, config(i))
    end

    @test indices == 1:dim
    @test length(unique(configurations)) == dim

    for rep=1:30
        rand!(s)
        for i=1:length(config(s))
            fl = length(first(config(s)))
            if i > fl
                v = first(config(s))[i-fl]
                old, new = flipat!(s, i)
                @test old == v
                @test new != v
                setat!(s, i, old)
                @test first(config(s))[i-fl] == v
            else
                v = last(config(s))[i]
                old, new = flipat!(s, i)
                @test old == v
                @test new != v
                setat!(s, i, old)
                @test last(config(s))[i] == v
            end
        end
    end
end

@testset "NAryState" begin
    s = NAryState(3, 4)
    test_state_properties(s, 3, 4, Float32, AbstractArray)
    test_state_dyn_single(s)
end

@testset "Double-NAryState" begin
    s = DoubleState(NAryState(3, 4))
    test_state_properties(s, 3, 4*2, Float32, Tuple{AbstractArray, AbstractArray})
    test_state_dyn_double(s)
end

@testset "WrappedChanges : NAryState" begin
    sb = NAryState(3, 4)
    s = ModifiedState(sb)
    test_state_properties(s, 3, 4, Float32, AbstractArray)
    test_state_dyn_single(s)
end

@testset "WrappedChanges : Double-NAryState" begin
    sb = ModifiedState(NAryState(3, 4))
    s = DoubleState(sb)
    test_state_properties(s, 3, 4*2, Float32, Tuple{AbstractArray, AbstractArray})
    test_state_dyn_double(s)
end

@testset "LU: WrappedChanges : NAryState" begin
    sb = NAryState(3, 4)
    s = LUState(ModifiedState(sb), nothing)
    test_state_properties(s, 3, 4, Float32, AbstractArray)
    test_state_dyn_single(s)
end

@testset "LU: WrappedChanges : Double-NAryState" begin
    sb = ModifiedState(NAryState(3, 4))
    s = LUState(DoubleState(sb), nothing)
    test_state_properties(s, 3, 4*2, Float32, Tuple{AbstractArray, AbstractArray})
    test_state_dyn_double(s)
end
