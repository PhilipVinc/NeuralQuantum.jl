using NeuralQuantum
using Test
using NeuralQuantum: flipat!, flipat_fast!

function test_state_dyn_single(s)
    l = local_dimension(s)
    n = nsites(s)
    dim = spacedimension(s)

    rs = deepcopy(raw_state(s))

    configurations = []
    indices = []
    cf = zeros(eltype(s), l)
    for i=1:dim
        set!(s, i)
        for j=1:dim
            cf[j] = rand(0:(l-1))
            setat!( s, j, cf[j])
            setat!(rs, j, cf[j])
        end

    end
end

#=
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
=#
