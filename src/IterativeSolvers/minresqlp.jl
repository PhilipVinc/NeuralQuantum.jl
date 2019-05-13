module MinresQlp
export minresqlp_iterable, minresqlp, minresqlp!
using Printf
import IterativeSolvers: zerox, ConvergenceHistory, reserve!, nextiter!, setconv
import IterativeSolvers: shrink!
import LinearAlgebra: BLAS.axpy!, givensAlgorithm, norm, mul!, axpy!
import Base: iterate

mutable struct MINRESQLPIterable{matT, solT, vecT <: DenseVector, shiftT<: Number, realT <: Real}
    A::matT
    shift::shiftT
    skew_hermitian::Bool
    x::solT

    # Krylov basis vectors
    r1::vecT
    r2::vecT
    r3::vecT

    #
    w::vecT
    wl::vecT
    xl2::vecT
    wl2::vecT
    v::vecT

    #flags
    flag0::Int
    flag::Int
    iters::Int
    QLPiter::Int
    beta::realT
    tau::realT
    taul::realT

    phi::realT
    beta1::realT #almost useless
    betan::realT
    gmin::realT
    cs::realT
    sn::realT
    cr1::realT
    sr1::realT
    cr2::realT
    sr2::realT
    dltan::realT
    eplnn::realT
    gama::realT
    gamal::realT
    gamal2::realT
    gama_QLP::realT
    gamal_QLP::realT
    eta::realT
    etal::realT
    etal2::realT
    vepln::realT
    veplnl::realT
    veplnl2::realT
    vepln_QLP::realT
    ul3::realT
    ul2::realT
    ul::realT
    u::realT
    ul_QLP::realT
    u_QLP::realT

    rnorm::realT
    Arnorm::realT
    xnorm::realT
    relAres::realT

    xl2norm::realT
    Axnorm::realT
    Anorm::realT
    Acond::realT
    relres::realT
    gminl::realT

    noprecon::Bool

    #lims
    TranCond::realT
    maxxnorm::realT
    Acondlim::realT

    # Bookkeeping
    mv_products::Int
    maxiter::Int
    tolerance::realT
    resnorm::realT
end

function minresqlp_iterable!(x, A, b;
    initially_zero::Bool = false,
    skew_hermitian::Bool = false,
    tol = sqrt(eps(real(eltype(b)))),
    maxiter = size(A, 2),
    shift   = 0.0,
    M       = nothing,
    TranCond=10e6,
    maxxnorm=10e6,
    Acondlim=10e14)

    T = eltype(x)
    rT= real(T)
    HessenbergT = skew_hermitian ? T : real(T)

    n = length(b)
    r1 = similar(b);    r1.=real(T)(0)
    r2 = similar(b);    copyto!(r2, b)
    r3 = similar(b);    copyto!(r3, b)
    beta1 = norm(r2)

    if isa(M, Nothing)
        noprecon = true
    else
        noprecon = false
        precond!(r3, M, r2)
        beta1 = real(r3'*r2)  #beta1 = r3.T.dot(r2) #teta
        if beta1 < 0.0
            print("Error: M is indefinite!")
        else
            beta1 = sqrt(beta1)
        end
    end
    relres = beta1/(beta1+1e-50)

    #x   = zeros(T, n)
    w   = zeros(T, n)
    wl  = zeros(T, n)
    xl2 = zeros(T, n)
    wl2 = zeros(T, n)
    v   = zeros(T, n)

    MINRESQLPIterable{typeof(A), typeof(x), typeof(b), typeof(shift), rT}(
        A, shift, skew_hermitian, x,
        r1, r2, r3,
        w, wl, xl2, wl2, v,
        -2, -2, 0, 0, zero(rT), zero(rT), zero(rT),   # flag0, flag, iters, qlpiter, beta, tau, taul
        beta1, beta1, beta1, zero(rT), -one(rT), zero(rT), -one(rT), zero(rT), #phi, beta1, betan, gmin, cs, sn, cr1, sr1
        -one(rT), zero(rT), zero(rT), zero(rT),  #cr2, sr2, dltan, eplnn,
        zero(rT), zero(rT), zero(rT), zero(rT), zero(rT), #gama, gamal, gamal2, gama_QLP, gamal_QLP
        zero(rT), zero(rT), zero(rT), zero(rT), zero(rT), zero(rT), zero(rT), #eta etal etal2 vepln veplnl veplnl2, vepln_QLP
        zero(rT), zero(rT), zero(rT), zero(rT), zero(rT), zero(rT), # ul3, ul2 ul u ul_QLP u_QLP
        beta1, zero(rT), zero(rT), zero(rT), # rnorm xnorm Arnorm relAres
        zero(rT), zero(rT), zero(rT), one(rT), #xl2norm axnorm anorm acond
        relres, zero(rT), #relres gminl,
        noprecon,
        TranCond, maxxnorm, TranCond,
        0, maxiter, tol, rT(123)
    )
end

converged(m::MINRESQLPIterable) = m.resnorm ≤ m.tolerance || m.flag != m.flag0

start(::MINRESQLPIterable) = 1

done(m::MINRESQLPIterable, iteration::Int) = iteration > m.maxiter || converged(m)

function iterate(m::MINRESQLPIterable, iteration::Int=start(m))
    if done(m, iteration) return nothing end

    m.iters += 1
    betal = m.beta
    m.beta = m.betan
    m.v .= m.r3./m.beta        # v = r3/beta
    #r3 = A*v            # r3 = Ax(A, v)
    mul!(m.r3, m.A, m.v)

    m.shift != 0.0 && axpy!(-m.shift, m.v, m.r3) # r3 .= r3 - shift*v
    #TODO Fix right mult
    m.iters > 1 && axpy!(-m.beta/betal, m.r1, m.r3) #r3 = r3 - r1*beta/betal

    alfa = real(m.r3'*m.v)                          # alfa = np.real(r3.T.dot(v))
    axpy!(-alfa/m.beta, m.r2, m.r3)   # r3 = r3 - r2*alfa/beta #TODO Fix right mult
    copyto!(m.r1, m.r2)                             # r1 = r2
    copyto!(m.r2, m.r3)                             # r2 = r3

    if m.noprecon
        m.betan = norm(m.r3)
        if m.iters == 1
            if m.betan == 0.0
                if m.alfa == 0.0
                    m.flag = 0
                    return
                else
                    m.flag = -1
                    mul!(m.x, m.b, inv(alfa)) #x = b/alfa
                    return
                end
            end
        end
    else#=
        precond!(r3, M, r2)
        m.betan = real(r2'*r3)
        if m.betan > 0.0
            m.betan = sqrt(m.betan)
        else
            print("Error: M is indefinite or singular!")
        end=#
        @error "Not Implemented"
    end
    pnorm = sqrt(betal^2 + alfa^2 + m.betan^2)

    #previous left rotation Q_{k-1}
    dbar = m.dltan
    dlta = m.cs*dbar + m.sn*alfa
    epln = m.eplnn
    gbar = m.sn*dbar - m.cs*alfa
    m.eplnn = m.sn*m.betan
    m.dltan = -m.cs*m.betan
    dlta_QLP = dlta
    #current left plane rotation Q_k
    gamal3 = m.gamal2
    m.gamal2 = m.gamal
    m.gamal = m.gama
    m.cs, m.sn, m.gama = SymGivens(gbar, m.betan)
    gama_tmp = m.gama
    taul2 = m.taul
    m.taul = m.tau
    m.tau = m.cs*m.phi
    m.Axnorm = sqrt(m.Axnorm^2 + m.tau^2)
    m.phi = m.sn*m.phi

    #previous right plane rotation P_{k-2,k}
    if m.iters > 2
        m.veplnl2 = m.veplnl
        m.etal2 = m.etal
        m.etal = m.eta
        dlta_tmp = m.sr2*m.vepln - m.cr2*dlta
        m.veplnl = m.cr2*m.vepln + m.sr2*dlta
        dlta = dlta_tmp
        m.eta = m.sr2*m.gama
        m.gama = -m.cr2 *m.gama
    end
    #current right plane rotation P{k-1,k}
    if m.iters > 1
        m.cr1, m.sr1, m.gamal = SymGivens(m.gamal, dlta)
        m.vepln = m.sr1*m.gama
        m.gama = -m.cr1*m.gama
    end

    #327
    #update xnorm
    xnorml = m.xnorm
    ul4 = m.ul3
    m.ul3 = m.ul2
    if m.iters > 2
        m.ul2 = (taul2 - m.etal2*ul4 - m.veplnl2*m.ul3)/m.gamal2
    end
    if m.iters > 1
        m.ul = (m.taul - m.etal*m.ul3 - m.veplnl * m.ul2)/m.gamal
    end
    xnorm_tmp = sqrt(m.xl2norm^2 + m.ul2^2 + m.ul^2)
    if abs(m.gama) > floatmin(real(eltype(m.A))) && xnorm_tmp < m.maxxnorm
        m.u = (m.tau - m.eta*m.ul2 - m.vepln*m.ul)/m.gama
        if sqrt(xnorm_tmp^2 + m.u^2) > m.maxxnorm
            m.u = 0
            m.flag = 6
        end
    else
        m.u = 0
        m.flag = 9
    end
    m.xl2norm = sqrt(m.xl2norm^2 + m.ul2^2)
    m.xnorm = sqrt(m.xl2norm^2 + m.ul^2 + m.u^2)

    #update w&x
    #Minres
    if (m.Acond < m.TranCond) && m.flag != m.flag0 && m.QLPiter == 0
        copyto!(m.wl2, m.wl)  # wl2 = wl
        copyto!(m.wl, m.w)    # wl = w
        m.w .= (m.v .- epln .* m.wl2 .- dlta_QLP .* m.wl)./gama_tmp
        if m.xnorm < m.maxxnorm
            m.x .+= m.tau .* m.w
        else
            m.flag = 6
        end
    #Minres-QLP
    else
        m.QLPiter += 1
        if m.QLPiter == 1
            if (m.iters > 1)    # construct w_{k-3}, w_{k-2}, w_{k-1}
                if m.iters > 3
                    m.wl2 .= gamal3.*m.wl2 .+ veplnl2.*m.wl .+ etal.*w
                end
                if m.iters > 2
                    m.wl .= gamal_QLP.*m.wl .+ vepln_QLP.*w
                end
                lmul!(gama_QLP, w)          #w = gama_QLP*w
                xl2 .= x .- m.wl.*ul_QLP .- w.*u_QLP
            end
        end

        # 369
        if m.iters == 1
            m.wl2 .= m.wl
            m.wl .=    m.v .* m.sr1
            m.w    .= .- m.v .* m.cr1
        elseif m.iters == 2
            m.wl2 .= m.wl
            m.wl  .= m.w .* m.cr1 .+ m.v .* m.sr1
            m.w   .= m.w .* m.sr1 .- m.v .* m.cr1
        else
            m.wl2 .= m.wl
            m.wl  .= m.w
            m.w   .= m.wl2 .* m.sr2 .- m.v .* m.cr2
            m.wl2 .= m.wl2 .* m.cr2 .+ m.v .* m.sr2
            m.v   .= m.wl  .* m.cr1 .+ m.w .* m.sr1
            m.w   .= m.wl  .* m.sr1 .- m.w .* m.cr1
            m.wl  .= m.v
        end
        m.xl2 .= m.xl2 .+ m.wl2 .* m.ul2
        m.x   .= m.xl2 .+ m.wl  .* m.ul .+ m.w .* m.u
    end

    # 388
    #next right plane rotation P{k-1,k+1}
    gamal_tmp = m.gamal
    m.cr2, m.sr2, m.gamal = SymGivens(m.gamal, m.eplnn)
    #transfering from Minres to Minres-QLP
    m.gamal_QLP = gamal_tmp
    m.vepln_QLP = m.vepln
    m.gama_QLP = m.gama
    m.ul_QLP = m.ul
    m.u_QLP = m.u
    ## Estimate various norms
    abs_gama = abs(m.gama)
    Anorml = m.Anorm
    m.Anorm = max(m.Anorm, pnorm, m.gamal, abs_gama)  #Anorm = maximum([Anorm, pnorm, gamal, abs_gama])
    if m.iters == 1
        m.gmin = m.gama
        m.gminl = m.gmin
    elseif m.iters > 1
        gminl2 = m.gminl
        m.gminl = m.gmin
        gmin = min(gminl2, m.gamal, abs_gama)   #gmin = minimum([gminl2, gamal, abs_gama])
    end

    #409
    Acondl = m.Acond
    m.Acond = m.Anorm / m.gmin
    rnorml = m.rnorm
    relresl = m.relres
    if m.flag != 9
        m.rnorm = m.phi
    end
    m.relres = m.rnorm / (m.Anorm * m.xnorm + m.beta1)
    rootl = sqrt(gbar ^2 + m.dltan ^ 2)
    m.Arnorm = rnorml * rootl
    m.relAres = rootl / m.Anorm
    ## See if any of the stopping criteria are satisfied.
    epsx = m.Anorm * m.xnorm * eps(typeof(m.xnorm))

    if (m.flag == m.flag0) || (m.flag == 9)
        t1 = 1 + m.relres
        t2 = 1 + m.relAres
        if m.iters >= m.maxiter
            m.flag = 8 #exit before maxit
        end
        if m.Acond >= m.Acondlim
            m.flag = 7 #Huge Acond
        end
        if m.xnorm >= m.maxxnorm
            m.flag = 6 #xnorm exceeded
        end
        if epsx >= m.beta1
            m.flag = 5 #x = eigenvector
        end
        if t2 <= 1
            m.flag = 4 #Accurate Least Square Solution
        end
        if t1 <= 1
            m.flag = 3 #Accurate Ax = b Solution
        end
        if m.relAres <= m.tolerance
            m.flag = 2 #Trustful Least Square Solution
        end
        if m.relres <= m.tolerance
            m.flag = 1 #Trustful Ax = b Solution
        end
    end
    if m.flag == 2 || m.flag == 4 || m.flag == 6 || m.flag == 7
        #possibly singular
        m.iters = m.iters - 1
        m.Acond = Acondl
        m.rnorm = rnorml
        m.relres = relresl
        #else show...
    end


    # The approximate residual is cheaply available
    m.resnorm = min(m.relres, m.relAres)

    min(m.relres, m.relAres), m.iters
end

function complete!(m::MINRESQLPIterable)
    Miter = m.iters - m.QLPiter

    mul!(m.r1, m.A, m.x)
    axpby!(m.shift, m.x, -1.0, m.r1)
    m.r1 .+= m.b

    m.rnorm = norm(m.r1)
    Arnorm = norm(m.A*m.r1 - m.shift*m.r1)
    m.xnorm = norm(m.x)
    relres = m.rnorm/(m.Anorm*m.xnorm + m.beta1)
    relAres = 0
    if rnorm > floatmin(real(eltype(A)))
        relAres = Arnorm/(Anorm*rnorm)
    end
end
"""
    minresqlp!(x, A, b; kwargs...) -> x, [history]

"""
function minresqlp!(x, A, b;
    skew_hermitian::Bool = false,
    verbose::Bool = false,
    log::Bool = false,
    tol = sqrt(eps(real(eltype(b)))),
    maxiter::Int = size(A, 2),
    initially_zero::Bool = false
)
    history = ConvergenceHistory(partial = !log)
    history[:tol] = tol
    log && reserve!(history, :resnorm, maxiter)
    log && reserve!(history, :rnorm, maxiter)
    log && reserve!(history, :Arnorm, maxiter)
    log && reserve!(history, :relres, maxiter)
    log && reserve!(history, :relAres, maxiter)
    log && reserve!(history, :Anorm, maxiter)
    log && reserve!(history, :Acond, maxiter)
    log && reserve!(history, :xnorm, maxiter)

    iterable = minresqlp_iterable!(x, A, b;
        skew_hermitian = skew_hermitian,
        tol = tol,
        maxiter = maxiter,
        initially_zero = initially_zero
    )

    if log
        history.mvps = iterable.mv_products
    end

    verbose && println("     iter     rnorm    Arnorm   relres   " *
                        "relAres    Anorm    Acond    xnorm")
    for (iteration, resnorm) = enumerate(iterable)
        m=iterable
        if log
            nextiter!(history, mvps = 1)
            push!(history, :resnorm, resnorm)
            push!(history, :rnorm, m.rnorm)
            push!(history, :Arnorm, m.Arnorm)
            push!(history, :relres, m.relres)
            push!(history, :relAres, m.relAres)
            push!(history, :Anorm, m.Anorm)
            push!(history, :Acond, m.Acond)
            push!(history, :xnorm, m.xnorm)
        end
        #verbose && @printf("%3d\t%1.2e\n", iteration, resnorm)
        verbose && @printf("%8g    %8.2e %8.2eD %8.2e %8.2eD %8.2e %8.2e  %8.2e\n",
                           m.iters, m.rnorm, m.Arnorm, m.relres, m.relAres, m.Anorm, #should be m.Anorml
                           m.Acond, m.xnorm)
    end

    verbose && println()
    log && setconv(history, converged(iterable))
    log && shrink!(history)

    log ? (iterable.x, history) : iterable.x
end

"""
    minres(A, b; kwargs...) -> x, [history]

Same as [`minres!`](@ref), but allocates a solution vector `x` initialized with zeros.
"""
minresqlp(A, b; kwargs...) = minresqlp!(zerox(A, b), A, b; initially_zero = true, kwargs...)



### ADDD
function SymGivens(a, b)
    if b == 0
        if a == 0
            c = 1
        else
            c = sign(a)
        end
        s = 0
        r = abs(a)
    elseif a == 0
        c = 0
        s = sign(b)
        r = abs(b)
    elseif abs(b) > abs(a)
        t = a / b
        s = sign(b) / sqrt(1 + t ^ 2)
        c = s * t
        r = b / s
    else
        t = b / a
        c = sign(a) / sqrt(1 + t ^ 2)
        s = c * t
        r = a / c
    end

    return c, s, r
end

end
