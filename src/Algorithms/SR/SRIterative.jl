struct SRIterativeCache{Tc,T,G,E,TT}
    S::T
    S_c::Tc
    F::G
    ∇out::E

    time_evolving::TT
end

function _sr_iterative_cache(algo::SR, prob, net, par_cache)
    T = eltype(trainable_first(net))
    g = grad_cache(T, net)
    gv = vec_data(g)
    if algo.use_fullmat
        S = tuple([similar(gv, T, length(gv), length(gv))
                        for gv in vec_data(g)]...)
        if T isa Real
            Sc = tuple([similar(gv, Complex{T}, length(gv), length(gv))
                            for gv in vec_data(g)]...)
        else
            Sc = S
        end
    else # use O
        S = tuple([SrMatrix(T, similar(gv, T, length(gv), 1), 0, par_cache)
                    for gv in vec_data(g)]...)
        Sc = tuple([nothing for gv in vec_data(g)]...)
    end
    return SRIterativeCache(S, Sc, g, grad_cache(T, net),
                            algo.time_evo)
end

function setup_algorithm!(g::SRIterativeCache, ∇C, Ô, par_cache)
    α = g.time_evolving isa Val{true} ? true * im : true

    O = Ô
    for (Sc, S, F) in zip(g.S_c, g.S, vec_data(g.F))
        if S isa SrMat
            init!(S, O)
            if eltype(F) <: Real
                F .= real.(α .* ∇C)
            else
                F .= α .* ∇C
            end
        else
            T = eltype(S)
            N = size(Ô, 2)

            # TODO MPI Sync Sc
            mul!(Sc, O, O')

            if T <: Real
                S .= real.(Sc ./ N)
                F .= real.(α  .* ∇C)
            else
                # We take the conjugate because O is actually Oconj matrix respect
                # to the standard SR implementation..
                S .= conj.(Sc) ./ N
                F .= α .* ∇C
            end

            workers_mean!(S, par_cache)
        end
    end

end

function precondition!(data::SRIterativeCache, params::SR, iter_n)
    return _precondition!(data, params, iter_n)
end

function _precondition!(data::SRIterativeCache, params::SR, iter_n)
    ϵ = params.sr_diag_shift
    success = true

    for (Δw, S, F) in zip(vec_data(data.∇out), data.S, vec_data(data.F))
        # new matrix
        if params.precondition_type == sr_none
            Sprecond = S
        elseif params.precondition_type == sr_shift
            Sprecond = S + convert(eltype(S), ϵ) * I
        elseif params.precondition_type == sr_multiplicative
            λ0 = params.λ0
            b = params.b
            λmin = params.λmin
            #vv = eigvals(S)
            #println("regol $(max(λ0*b^iter_n, λmin)) --> $(minimum(vv)), a $(maximum(vv))")
            λ = convert(eltype(S), max(λ0 * b^iter_n, λmin))
            Sprecond = S + λ * Diagonal(diag(S))

        end

        if params.algorithm == sr_qlp
            x, hist = minresqlp(
                Sprecond,
                F,
                maxiter = size(S, 2) * 10,
                log = true,
                verbose = false,
                tol = params.sr_precision,
            )
        elseif params.algorithm == sr_minres
            x, hist = minres(
                Sprecond,
                F,
                maxiter = size(S, 2) * 10,
                log = true,
                verbose = false,
                tol = params.sr_precision)
        elseif params.algorithm == sr_lsq
            x, hist = lsqr(
                Sprecond,
                F,
                maxiter = size(S, 2) * 10,
                log = true,
                verbose = false,
            )
        elseif params.algorithm == sr_cg
            x, hist = cg(
                Sprecond,
                F,
                maxiter = size(S, 2) * 10,
                log = true,
                verbose = false,
                tol = params.sr_precision,
            )
        end
            #x, hist = cg(S.+ ϵ*I, F, maxiter=size(S,2)*10, log=true, verbose=true, tol=10e-10)
        if eltype(Δw) <: Real
            Δw .= real.(x)
        else
            Δw .= x
        end
        add_iters = 1
        while !hist.isconverged
            println("minresqlp not conerged. Additional $(size(S,2)*10) iters for the $add_iters time.")
            x, hist = minresqlp(
                Δw,
                Sprecond,
                F,
                maxiter = size(S, 2) * 10,
                log = true,
            )
            Δw .= x
            add_iters += 1
            add_iters > 5 && break
        end
        if add_iters > 5 && !hist.isconverged
            success = false
            Δw .= 0.0
        end
    end
    return data.∇out
end
