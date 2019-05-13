using NeuralQuantumBase
using Test
using NeuralQuantumBase: LdagL_L_prob, LdagL_Lmat_prob
Nsites = 4
T = Float64

prob_types = [LdagL_sop_prob, LdagL_spmat_prob]
@testset "LdagL problem: $prob_T" for prob_T=prob_types
    lind = quantum_ising_lind(SquareLattice([Nsites],PBC=true), g=1.6, V=2.0, γ=1.0)
    net  = cached(rNDM(T, Nsites, 2, 1))

    prob = prob_T(T, lind);

    v    = state(prob, net)

    L=liouvillian(lind)
    LdagL = L.data'*L.data
    rho   = dm(net, prob, false).data
    rhov  = vec(rho)

    ic = NeuralQuantumBase.MCMCSREvaluationCache(net); zero!(ic)

    Clocs_ex  = (LdagL*rhov)./rhov
    for i=1:spacedimension(v)
        set_index!(v, i)
        NeuralQuantumBase.sample_network!(ic, prob, net, v)
    end
    Clocs_net_S = ic.Evalues

    @test Clocs_ex ≈ Clocs_net_S
end

Nsites = 4
T = Float64

prob_types = [LdagL_L_prob, LdagL_Lmat_prob]
@testset "LdagL problem: $prob_T" for prob_T=prob_types
    lind = quantum_ising_lind(SquareLattice([Nsites],PBC=true), g=1.6, V=2.0, γ=1.0)
    net  = cached(rNDM(T, Nsites, 2, 1))
    net  = (rNDM(T, Nsites, 2, 1))

    prob = NeuralQuantumBase.LdagL_sop_prob(T, lind);
    probL = prob_T(T, lind);

    v    = state(prob, net)

    L=liouvillian(lind)
    LdagL = L.data'*L.data
    rho   = dm(net, prob, false).data
    rhov  = vec(rho)

    ic = NeuralQuantumBase.MCMCSREvaluationCache(net); zero!(ic)
    icL = NeuralQuantumBase.MCMCSRLEvaluationCache(net); zero!(icL)

    Clocs_ex  = (LdagL*rhov)./rhov
    for i=1:spacedimension(v)
        set_index!(v, i)
        NeuralQuantumBase.sample_network_wholespace!(ic, prob, net, v)
        NeuralQuantumBase.sample_network_wholespace!(icL, probL, net, v)
    end

    SREval = EvaluatedNetwork(SR(), net)
    SREvalL = EvaluatedNetwork(SR(), net)

    evaluation_post_sampling!(SREval, ic)
    evaluation_post_sampling!(SREvalL, icL)

    @test all([l≈r for (l,r)=zip(SREval.F, SREvalL.F)])
end

# random
prob_types = [LdagL_sop_prob, LdagL_spmat_prob]
@testset "LdagL random problem: $prob_T" for prob_T=prob_types
    Nsites=2
    lind = quantum_ising_lind(SquareLattice([Nsites],PBC=false), g=1.6, V=2.0, γ=1.0)
    net  = cached(rNDM(T, Nsites, 2, 1))

    Ham  = SparseOperator(lind.H)
    Ham.data.=sprand(ComplexF64, size(Ham.data,1),size(Ham.data,2),0.5)
    cops = jump_operators(lind)
    prob = prob_T(T, Ham, cops);

    v    = state(prob, net)

    L=liouvillian(Ham, cops)
    LdagL = L.data'*L.data
    rho   = dm(net, prob, false).data
    rhov  = vec(rho)

    ic = NeuralQuantumBase.MCMCSREvaluationCache(net); zero!(ic)

    Clocs_ex  = (LdagL*rhov)./rhov
    for i=1:spacedimension(v)
        set_index!(v, i)
        NeuralQuantumBase.sample_network!(ic, prob, net, v)
    end
    Clocs_net_S = ic.Evalues

    @test Clocs_ex ≈ Clocs_net_S
end
