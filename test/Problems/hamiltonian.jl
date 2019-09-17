using NeuralQuantum
using Test
using NeuralQuantum: LdagLSparseSuperopProblem, LRhoSparseOpProblem, LRhoKLocalOpProblem
using NeuralQuantum: init_lut!, HamiltonianGSEnergyProblem

Nsites = 4
T = Float64
lind = quantum_ising_lind(SquareLattice([Nsites],PBC=true), g=1.3, V=2.0, γ=1.0)
ham  = lind.H

net  = cached(RBM(T, Nsites, 2))

prob_sp = HamiltonianGSEnergyProblem(T, ham, operators=false);
prob_op = HamiltonianGSEnergyProblem(T, ham, operators=true);

v    = state(prob_sp, net)

H = DenseOperator(ham).data
ψ = ket(net, prob_sp, false).data
Clocs_ex = (H*ψ)./ψ

ic_sp   = NeuralQuantum.MCMCSREvaluationCache(net, prob_sp);  zero!(ic_sp)
ic_op   = NeuralQuantum.MCMCSREvaluationCache(net, prob_op);  zero!(ic_op)

for i=1:spacedimension(v)
    set_index!(v, i)
    NeuralQuantum.sample_network!(ic_sp, prob_sp, net, v);
    NeuralQuantum.sample_network!(ic_op, prob_op, net, v);
end

SREval_sp    = EvaluatedNetwork(SR(), net)
SREval_op    = EvaluatedNetwork(SR(), net)

evaluation_post_sampling!(SREval_sp,    ic_sp)
evaluation_post_sampling!(SREval_op,    ic_op)

@testset "LookUp table evaluation" begin
    @test ic_op.Evalues ≈ Clocs_ex
    @test ic_sp.Evalues ≈ Clocs_ex

    @test SREval_op.L ≈ SREval_sp.L
    @test ic_op.Evalues ≈ ic_sp.Evalues
    @test all([l≈r for (l,r)=zip(SREval_op.F, SREval_sp.F)])
    @test all([l≈r for (l,r)=zip(SREval_op.S, SREval_sp.S)])
end
