struct SRDirectCache{Tc,T,G,E,TT}
    S_c::Tc
    S::T
    F::G
    ∇out::E

    time_evolving::TT
end

function _sr_direct_cache(algo::SR, prob, net, par_cache)
    T = eltype(trainable_first(net))
    g = grad_cache(T, net)
    gv = vec_data(g)

    S = [similar(gv, T, length(gv), length(gv)) for gv in vec_data(g)]
    if T <: Real
        Sc = [similar(gv, Complex{T}, length(gv), length(gv)) for gv in vec_data(g)]
    else # is real
        Sc = S
    end

    return SRDirectCache(Sc, S, g, grad_cache(T, net),
                         algo.time_evo)
end

function setup_algorithm!(g::SRDirectCache, ∇C, Ô, par_cache)
    α = g.time_evolving isa Val{true} ? true * im : true

    O = Ô
    for (Sc, S, F) in zip(g.S_c, g.S, vec_data(g.F))
        T = eltype(S)
        N = size(Ô, 2)

        mul!(Sc, O, O')

        if T <: Real
            S .= real.(Sc ./ N)
            F .= real.(α .* ∇C)
        else
            # We take the conjugate because O is actually Oconj matrix respect
            # to the standard SR implementation..
            S .= conj.(Sc) ./ N
            F .= α .* ∇C
        end

        workers_sum!(S, par_cache)
    end
end


function precondition!(data::SRDirectCache, params::SR, iter_n)
    ϵ = params.sr_diag_shift

    success = true

    for (Δw, S, F) in zip(vec_data(data.∇out), data.S, vec_data(data.F))
        # new matrix
        if params.precondition_type == sr_none
            Sprecond = S
        elseif params.precondition_type == sr_shift
            #Sprecond = S + convert(eltype(S), ϵ)*I
            @inbounds @simd for i = 1:size(S, 1)
                S[i, i] += convert(eltype(S), ϵ)
            end
            Sprecond = S
        elseif params.precondition_type == sr_multiplicative
            λ0 = params.λ0
            b = params.b
            λmin = params.λmin
            λ = convert(eltype(S), max(λ0 * b^iter_n, λmin))
            Sprecond = S + λ * Diagonal(diag(S))
        end

        @assert is_iterative(params) == false
        if params.algorithm == sr_diag
            Sinv = pinv(Sprecond)
            mul!(Δw, Sinv, F)
        elseif params.algorithm == sr_cholesky
            C = cholesky!(Hermitian(S), check = true)
            copyto!(Δw, F)
            ldiv!(C, Δw)
        elseif params.algorithm == sr_div
            copyto!(Δw, F)
            ldiv!(Hermitian(S), Δw)
        else
            throw("Alg not known")
        end
    end
    return data.∇out
end
###########
