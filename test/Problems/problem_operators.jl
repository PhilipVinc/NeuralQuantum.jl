using NeuralQuantum
using Test
using NeuralQuantum: LdagL_L_prob, LdagL_Lmat_prob, LdagL_Lrho_op_prob
Nsites = 4
T = Float64

@testset "LdagL problem: LdagL_Lrho_op_prob" begin
    lind = quantum_ising_lind(SquareLattice([Nsites],PBC=true), g=1.6, V=2.0, γ=0.0)
    net  = cached(rNDM(T, Nsites, 2, 1))

    prob = LdagL_L_prob(T, lind);
    probL = LdagL_Lrho_op_prob(T, lind);

    v    = state(prob, net)

    L=liouvillian(lind)
    LdagL = L.data'*L.data
    rho   = dm(net, prob, false).data
    rhov  = vec(rho)

    ic = NeuralQuantum.MCMCSRLEvaluationCache(net, prob); zero!(ic)
    icL = NeuralQuantum.MCMCSRLEvaluationCache(net, probL); zero!(icL)

    Clocs_ex  = abs.((L.data*rhov)./rhov).^2
    for i=1:spacedimension(v)
        set_index!(v, i)
        NeuralQuantum.sample_network!(ic, prob, net, v)
        NeuralQuantum.sample_network!(icL, probL, net, v)
    end

    SREval = EvaluatedNetwork(SR(), net)
    SREvalL = EvaluatedNetwork(SR(), net)

    evaluation_post_sampling!(SREval, ic)
    evaluation_post_sampling!(SREvalL, icL)

    @test SREval.L ≈ SREvalL.L
    @test ic.Evalues ≈ icL.Evalues
    @test all([l≈r for (l,r)=zip(SREval.F, SREvalL.F)])
    @test all([l≈r for (l,r)=zip(SREval.S, SREvalL.S)])
end
