using NeuralQuantum
using Test
using NeuralQuantum: LdagL_L_prob, LdagL_Lmat_prob, LdagL_Lrho_op_prob
using NeuralQuantum: init_lut!
Nsites = 4
T = Float64

lind = quantum_ising_lind(SquareLattice([Nsites],PBC=true), g=1.3, V=2.0, γ=1.0)


@testset "LdagL (operator) problem with NDM: LdagL_Lrho_op_prob" begin
net  = cached(NDM(T, Nsites, 2, 1))

prob = LdagL_Lmat_prob(T, lind);
probL = LdagL_Lrho_op_prob(T, lind);

v    = state(prob, net)
vL    = state(probL, net)

L=liouvillian(lind)
LdagL = L.data'*L.data
rho   = dm(net, prob, false).data
rhov  = vec(rho)

ic     = NeuralQuantum.MCMCSRLEvaluationCache(net, prob);  zero!(ic)
icL_fb = NeuralQuantum.MCMCSRLEvaluationCache(net, prob);  zero!(icL_fb)
icL    = NeuralQuantum.MCMCSRLEvaluationCache(net, probL); zero!(icL)

Clocs_ex  = abs.((L.data*rhov)./rhov).^2
for i=1:spacedimension(v)
    set_index!(v, i)
    NeuralQuantum.sample_network!(ic,     prob,  net, v);
    NeuralQuantum.sample_network!(icL_fb, probL, net, v);
    NeuralQuantum.sample_network!(icL,    probL, net, vL);
end

SREval     = EvaluatedNetwork(SR(), net)
SREvalL    = EvaluatedNetwork(SR(), net)
SREvalL_fb = EvaluatedNetwork(SR(), net)

evaluation_post_sampling!(SREval,     ic)
evaluation_post_sampling!(SREvalL,    icL)
evaluation_post_sampling!(SREvalL_fb, icL_fb)

@testset "LookUp table evaluation" begin
    @test_broken SREval.L ≈ SREvalL.L
    @test_broken ic.Evalues ≈ icL.Evalues
    @test_broken all([l≈r for (l,r)=zip(SREval.F, SREvalL.F)])
    @test_broken all([l≈r for (l,r)=zip(SREval.S, SREvalL.S)])
end
@testset "Fallback (no LUT)" begin
    @test SREval.L ≈ SREvalL_fb.L
    @test ic.Evalues ≈ icL_fb.Evalues
    @test all([l≈r for (l,r)=zip(SREval.F, SREvalL_fb.F)])
    @test all([l≈r for (l,r)=zip(SREval.S, SREvalL_fb.S)])
end
end


@testset "LdagL (operator) problem with RBMSplit: LdagL_Lrho_op_prob" begin
net  = cached(RBMSplit(Complex{T}, Nsites, 2))

prob = LdagL_Lmat_prob(T, lind);
probL = LdagL_Lrho_op_prob(T, lind);

v    = state(prob, net)
vL    = state(probL, net)

L=liouvillian(lind)
LdagL = L.data'*L.data
rho   = dm(net, prob, false).data
rhov  = vec(rho)

ic     = NeuralQuantum.MCMCSRLEvaluationCache(net, prob);  zero!(ic)
icL_fb = NeuralQuantum.MCMCSRLEvaluationCache(net, prob);  zero!(icL_fb)
icL    = NeuralQuantum.MCMCSRLEvaluationCache(net, probL); zero!(icL)

Clocs_ex  = abs.((L.data*rhov)./rhov).^2
for i=1:spacedimension(v)
    set_index!(v, i)
    init_lut!(set_index!(vL, i), net)
    NeuralQuantum.sample_network!(ic,     prob,  net, v);
    NeuralQuantum.sample_network!(icL_fb, probL, net, v);
    NeuralQuantum.sample_network!(icL,    probL, net, vL);
end

SREval     = EvaluatedNetwork(SR(), net)
SREvalL    = EvaluatedNetwork(SR(), net)
SREvalL_fb = EvaluatedNetwork(SR(), net)

evaluation_post_sampling!(SREval,     ic)
evaluation_post_sampling!(SREvalL,    icL)
evaluation_post_sampling!(SREvalL_fb, icL_fb)

@testset "LookUp table evaluation" begin
    @test SREval.L ≈ SREvalL.L
    @test ic.Evalues ≈ icL.Evalues
    @test all([l≈r for (l,r)=zip(SREval.F, SREvalL.F)])
    @test all([l≈r for (l,r)=zip(SREval.S, SREvalL.S)])
end
@testset "Fallback (no LUT)" begin
    @test SREval.L ≈ SREvalL_fb.L
    @test ic.Evalues ≈ icL_fb.Evalues
    @test all([l≈r for (l,r)=zip(SREval.F, SREvalL_fb.F)])
    @test all([l≈r for (l,r)=zip(SREval.S, SREvalL_fb.S)])
end
end
