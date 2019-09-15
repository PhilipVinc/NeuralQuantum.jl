using NeuralQuantum, LinearAlgebra, Statistics
using Test, QuantumOptics
using NeuralQuantum: LdagL_L_prob, LdagL_Lmat_prob
Nsites = 4
T = Float64

@testset "Observables" begin
lattice = SquareLattice([Nsites],PBC=true)
lind = quantum_ising_lind(lattice, g=1.0, V=2.0, γ=1.0)
Sx  = QuantumLattices.LocalObservable(lind, sigmax, Nsites)
Sy  = QuantumLattices.LocalObservable(lind, sigmay, Nsites)
Sz  = QuantumLattices.LocalObservable(lind, sigmaz, Nsites)
H   = lind.H
oprob    = ObservablesProblem(Sx, Sy, Sz, H, operator=false)
oprob_op = ObservablesProblem(Sx, Sy, Sz, H, operator=true)
obs_dense = [DenseOperator(op).data for op=[Sx, Sy, Sz, H]]

net  = cached(RBMSplit(T, Nsites, 2))

sampl = MCMCSampler(Metropolis(), 3000, burn=50)
osampl = FullSumSampler()

v     = state(oprob, net)

ρ     = dm(net, oprob, false).data
#ex_val = [expect(o, ρ) for o=obs_dense]

ic    = NeuralQuantum.MCMCObsEvaluationCache(oprob)
ic_op = NeuralQuantum.MCMCObsEvaluationCache(oprob_op)

for i=1:spacedimension(v)
    set_index!(v, i)
    NeuralQuantum.sample_network!(ic, oprob, net, v)
    set_index!(v, i)
    NeuralQuantum.sample_network!(ic_op, oprob_op, net, v)
end

ObsEval     = EvaluatedNetwork(oprob, net)
ObsEval_op  = EvaluatedNetwork(oprob_op, net)

evaluation_post_sampling!(ObsEval,    ic)
evaluation_post_sampling!(ObsEval_op, ic_op)

@testset "Test Observables $i" for (i, obs)=enumerate(obs_dense)
    obs_ex = diag(ρ*obs)./diag(ρ)

    @test obs_ex ≈ ic.ObsVals[i]
end

@testset "Test Observables operators $i" for (i, obs)=enumerate(obs_dense)
    obs_ex = diag(ρ*obs)./diag(ρ)

    @test obs_ex ≈ ic_op.ObsVals[i]
end

end
