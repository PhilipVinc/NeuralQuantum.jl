struct GradientData{T} <: AlgorithmCache
    G::T
end

function algorithm_cache(algo::Gradient, prob, net, par_cache)
    T = eltype(trainable_first(net))
    g = grad_cache(T, net)
    return GradientData(g)
end

function setup_algorithm!(g::GradientData, data, par_cache)
    for field=vec_data(g.G)
        T = eltype(field)
        if T<: Real
            field .= real.(data)
        else
            field .= data
        end
    end
end

precondition!(g::GradientData, params::Gradient, iter_n) = g.G
