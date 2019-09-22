using NeuralQuantum
using Test
using NeuralQuantum: LdagLSparseSuperopProblem, LRhoSparseOpProblem
Nsites = 4
T = Float64

prob_types = [LdagLSparseSuperopProblem, LdagLSparseOpProblem]
@testset "LdagL problem: $prob_T" for prob_T=prob_types
    lind = quantum_ising_lind(SquareLattice([Nsites],PBC=true), g=1.6, V=2.0, γ=1.0)
    net  = cached(NDM(T, Nsites, 2, 1))

    prob = prob_T(T, lind);

    v    = state(prob, net)

    L=liouvillian(lind)
    LdagL = L.data'*L.data
    rho   = dm(net, prob, false).data
    rhov  = vec(rho)

    ic = NeuralQuantum.MCMCSREvaluationCache(net, prob); zero!(ic)

    Clocs_ex  = (LdagL*rhov)./rhov
    for i=1:spacedimension(v)
        set_index!(v, i)
        NeuralQuantum.sample_network!(ic, prob, net, v)
    end
    Clocs_net_S = ic.Evalues

    @test Clocs_ex ≈ Clocs_net_S
end

Nsites = 4
T = Float64

prob_types = [LRhoSparseSuperopProblem, LRhoSparseOpProblem]
@testset "LdagL problem: $prob_T" for prob_T=prob_types
    lind = quantum_ising_lind(SquareLattice([Nsites],PBC=true), g=1.6, V=2.0, γ=1.0)
    net  = cached(NDM(T, Nsites, 2, 1))
    net  = (NDM(T, Nsites, 2, 1))

    prob = NeuralQuantum.LdagLSparseSuperopProblem(T, lind);
    probL = prob_T(T, lind);

    v    = state(prob, net)

    L=liouvillian(lind)
    LdagL = L.data'*L.data
    rho   = dm(net, prob, false).data
    rhov  = vec(rho)

    ic = NeuralQuantum.MCMCSREvaluationCache(net, prob); zero!(ic)
    icL = NeuralQuantum.MCMCSRLEvaluationCache(net, probL); zero!(icL)

    Clocs_ex  = (LdagL*rhov)./rhov
    for i=1:spacedimension(v)
        set_index!(v, i)
        NeuralQuantum.sample_network_wholespace!(ic, prob, net, v)
        NeuralQuantum.sample_network_wholespace!(icL, probL, net, v)
    end

    SREval = EvaluatedNetwork(SR(), net)
    SREvalL = EvaluatedNetwork(SR(), net)

    evaluation_post_sampling!(SREval, ic)
    evaluation_post_sampling!(SREvalL, icL)

    @test all([l≈r for (l,r)=zip(SREval.F, SREvalL.F)])
end

@testset "LdagL problem: LRhoSparseOpProblem" begin
    lind = quantum_ising_lind(SquareLattice([Nsites],PBC=true), g=0.0, V=0.0, γ=1.0)
    net  = cached(NDM(T, Nsites, 2, 1))

    prob  = LRhoSparseSuperopProblem(T, lind);
    probL = LRhoSparseOpProblem(T, lind);

    v    = state(prob, net)

    L=liouvillian(lind)
    LdagL = L.data'*L.data
    rho   = dm(net, prob, false).data
    rhov  = vec(rho)

    ic  = NeuralQuantum.MCMCSRLEvaluationCache(net, prob); zero!(ic)
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

using SparseArrays, Random

# random
prob_types = [LdagLSparseSuperopProblem, LdagLSparseOpProblem]
@testset "LdagL random problem: $prob_T" for prob_T=prob_types
    Nsites=2
    lind = quantum_ising_lind(SquareLattice([Nsites],PBC=false), g=1.6, V=2.0, γ=1.0)
    net  = cached(NDM(T, Nsites, 2, 1))

    Ham  = SparseOperator(lind.H)
    Ham.data.=sprand(ComplexF64, size(Ham.data,1),size(Ham.data,2),0.5)
    Ham.data .= Ham.data + Ham.data'
    cops = jump_operators(lind)
    c1 = first(cops)
    c1.data = sprand(ComplexF64, size(c1.data,1),size(c1.data,2),0.1)
    prob = prob_T(T, Ham, cops);

    v    = state(prob, net)

    L=liouvillian(Ham, cops)
    LdagL = L.data'*L.data
    rho   = dm(net, prob, false).data
    rhov  = vec(rho)

    ic = NeuralQuantum.MCMCSREvaluationCache(net, prob); zero!(ic)

    Clocs_ex  = (LdagL*rhov)./rhov
    for i=1:spacedimension(v)
        set_index!(v, i)
        NeuralQuantum.sample_network!(ic, prob, net, v)
    end
    Clocs_net_S = ic.Evalues

    @test Clocs_ex ≈ Clocs_net_S
end

prob_types = [LRhoSparseSuperopProblem, LRhoSparseOpProblem]
@testset "LdagL random problem: $prob_T" for prob_T=prob_types
    Nsites=2
    lind = quantum_ising_lind(SquareLattice([Nsites],PBC=false), g=1.6, V=2.0, γ=1.0)
    net  = cached(NDM(T, Nsites, 2, 1))

    Ham  = SparseOperator(lind.H)
    Ham.data.=sprand(ComplexF64, size(Ham.data,1),size(Ham.data,2),0.5)
    Ham.data .= Ham.data + Ham.data'
    cops = jump_operators(lind)
    c1 = first(cops)
    c1.data = sprand(ComplexF64, size(c1.data,1),size(c1.data,2),0.1)
    prob = prob_T(T, Ham, cops);

    v    = state(prob, net)

    L=liouvillian(Ham, cops)
    rho   = dm(net, prob, false).data
    rhov  = vec(rho)

    ic = NeuralQuantum.MCMCSRLEvaluationCache(net, prob); zero!(ic)

    Clocs_ex  = abs.((L.data*rhov)./rhov).^2
    for i=1:spacedimension(v)
        set_index!(v, i)
        NeuralQuantum.sample_network!(ic, prob, net, v)
    end
    Clocs_net_S = ic.Evalues

    @test Clocs_ex ≈ Clocs_net_S
end
