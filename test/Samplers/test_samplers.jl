using NeuralQuantum
using NeuralQuantum: unsafe_get_el
using Test
using StatsBase: Histogram, fit
using HypothesisTests

function pdf(net, hilb)
    cnet = cached(net)
    ψ = ComplexF64.(Vector(cnet, hilb, false))
    probs = (abs2.(ψ)./sum(abs2.(ψ)))
    return probs
end

function sample_freqs(sampler, net, hilb)
    sampl = NeuralQuantum.SimpleIterativeSampler(net, sampler, hilb, batch_sz=16)
    NeuralQuantum.sample!(sampl)
    # extract indices
    ids = index(hilb, sampl.samples)

    hist = fit(Histogram, vec(ids), 1:spacedimension(hilb)+1)
    nhist = normalize(hist, mode=:pdf)

    return hist.weights
end

function test_sampler(sampler, net, hilb)
    exact_dist    = pdf(net, hilb)
    sampled_freqs = sample_freqs(sampler, net, hilb)
    tst = ChisqTest(sampled_freqs, exact_dist)
    @test pvalue(tst) >= 0.01
end

N = 4
n_samples = 1000
SEED = 123

lat  = HyperCube([N], periodic=true)
hilb = HomogeneousSpin(lat, 1//2)
H    = quantum_ising_hamiltonian(ComplexF32, lat, hilb, g=-1.0, V=2.0)

net = RBM(N, 1)
init_random_pars!(net, sigma=0.2^2)

n_samples = max(10000, 40*spacedimension(hilb))

@testset "Hamiltonian" begin
    @testset "ExactSampler" begin
        sampler = ExactSampler(n_samples, seed=SEED)
        test_sampler(sampler, net, hilb)
    end

    @testset "LocalRule" begin
        sampler = MetropolisSampler(LocalRule(), N*n_samples, 1, seed=SEED)
        test_sampler(sampler, net, hilb)

        sampler = MetropolisSampler(LocalRule(), n_samples, N+1, seed=SEED)
        test_sampler(sampler, net, hilb)
    end

    @testset "OperatorRule" begin
        sampler = MetropolisSampler(OperatorRule(H), n_samples, N+1, seed=SEED)
        test_sampler(sampler, net, hilb)
    end
end

@testset "Liouvillian" begin
    hilb2 = SuperOpSpace(hilb)
    net = NDM(N, 1, 1, af_logcosh)
    n_samples = max(N*10000, 40*spacedimension(hilb2))

    @testset "ExactSampler" begin
        sampler = ExactSampler(n_samples)
        test_sampler(sampler, net, hilb2)
    end

    @testset "LocalRule" begin
        sampler = MetropolisSampler(LocalRule(), n_samples, N+1, seed=SEED)
        test_sampler(sampler, net, hilb2)
    end

    @testset "NagyRule" begin
        sampler = MetropolisSampler(NagyRule(H), n_samples, N+1, seed=SEED)
        test_sampler(sampler, net, hilb2)
    end
end
