struct SRDirectCache{Tc,T,G,E}
    S_c::Tc
    S::T
    F::G
    ∇out::E
end

function _sr_direct_cache(algo::SR, prob, net)
    T  = eltype(trainable_first(net))
    g  = grad_cache(T, net)
    gv = vec_data(g)

    S  = [similar(gv, T, length(gv), length(gv)) for gv=vec_data(g)]
    if T isa Complex
        Sc = S
    else # is real
        Sc  = [similar(gv, Complex{T}, length(gv), length(gv)) for gv=vec_data(g)]
    end

    return SRDirectCache(Sc, S, g, grad_cache(T, net))
end

function setup_algorithm!(g::SRDirectCache, ∇C, Ô)
    O =  Ô
    for (Sc, S, F)=zip(g.S_c, g.S, vec_data(g.F))
        T = eltype(S)
        N = size(Ô,2)

        mul!(Sc, O, O')

        if T <: Real
            S  .= real.(Sc ./ N)
            F  .= real.(∇C)
        else
            Sc ./= N
            F  .= ∇C
        end
    end
end


function precondition!(data::SRDirectCache, params::SR, iter_n)
    ϵ = params.sr_diag_shift

    success = true

    for (Δw, S, F) = zip(vec_data(data.∇out), data.S, vec_data(data.F))
        # new matrix
        if params.precondition_type == sr_none
            Sprecond = S
        elseif params.precondition_type == sr_shift
            #Sprecond = S + convert(eltype(S), ϵ)*I
            @inbounds @simd for i=1:size(S,1)
                S[i,i] += convert(eltype(S), ϵ)
            end
            Sprecond = S
        elseif params.precondition_type == sr_multiplicative
            λ0 = params.λ0; b = params.b; λmin = params.λmin;
            λ = convert(eltype(S),max(λ0*b^iter_n, λmin))
            Sprecond = S + λ*Diagonal(diag(S))
        end

        @assert is_iterative(params) == false
        if params.algorithm == sr_diag
            Sinv = pinv(Sprecond)
            mul!(Δw, Sinv, F)
        elseif params.algorithm == sr_cholesky
            C = cholesky!(Hermitian(S))
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

struct SRIterativeCache{Tc,T,G,E}
    S_c::Tc
    S::T
    F::G
    ∇out::E
end

function _sr_iterative_cache(algo::SR, prob, net)
    T  = eltype(trainable_first(net))
    g  = grad_cache(T, net)
    gv = vec_data(g)
    S  = [similar(gv, T, length(gv), length(gv)) for gv=vec_data(g)]
    if T isa Real
        Sc  = [similar(gv, Complex{T}, length(gv), length(gv)) for gv=vec_data(g)]
    else
        Sc = S
    end
    return SRFullData(Sc, S, g, grad_cache(T, net))
end

function precondition!(data::SRIterativeCache, params::SR, iter_n)
    ϵ = params.sr_diag_shift

    success = true

    for (Δw, S, F) = zip(vec_data(data.∇out), data.S, vec_data(data.F))
        # new matrix
        if params.precondition_type == sr_none
            Sprecond = S
        elseif params.precondition_type == sr_shift
            Sprecond = S + convert(eltype(S), ϵ)*I
        elseif params.precondition_type == sr_multiplicative
            λ0 = params.λ0; b = params.b; λmin = params.λmin;
            #vv = eigvals(S)
            #println("regol $(max(λ0*b^iter_n, λmin)) --> $(minimum(vv)), a $(maximum(vv))")
            λ = convert(eltype(S),max(λ0*b^iter_n, λmin))
            Sprecond = S + λ*Diagonal(diag(S))

        end

        if !params.use_iterative
            try
                Δw = pinv(Sprecond)*F
            catch err
                println("Could not invert: $err")
                Δw = 0.0
                success = false
            end
        else
            x, hist = minresqlp(Sprecond, F, maxiter=size(S,2)*10, log=true, verbose=false, tol=params.sr_precision)
            #x, hist = cg(S.+ ϵ*I, F, maxiter=size(S,2)*10, log=true, verbose=true, tol=10e-10)
            if eltype(Δw) <: Real
                Δw .= real.(x)
            else
                Δw .= x
            end
            add_iters = 1
            while !hist.isconverged
                println("minresqlp not conerged. Additional $(size(S,2)*10) iters for the $add_iters time.")
                x, hist = minresqlp(ΔW, Sprecond, F, maxiter=size(S,2)*10, log=true)
                Δw .= x
                add_iters += 1
                add_iters > 5 && break
            end
            if add_iters > 5 && !hist.isconverged
                success = false
                Δw .= 0.0
            end
        end
    end
    return data.∇out
end
