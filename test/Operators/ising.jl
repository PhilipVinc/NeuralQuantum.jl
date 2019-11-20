using NeuralQuantum, QuantumOpticsBase
using LinearAlgebra, SparseArrays
using Test

N = 4
g = 0.7
V = 2.0

hilb = HomogeneousSpin(N)
hilbq = SpinBasis(1//2)^N

ops = []
opsq = []
H = LocalOperator(hilb)
Hq = DenseOperator(hilbq)
for i=1:N
    global H += g/2.0 * sigmax(hilb, i)
    global Hq += g/2.0 * embed(hilbq, i, sigmax(SpinBasis(1//2)))
    push!(ops, sigmam(hilb, i))
    push!(opsq, embed(hilbq, i, sigmam(SpinBasis(1//2))))
end

for i=1:N
    global H  += V/4.0 * sigmaz(hilb, i) * sigmaz(hilb, mod(i, N)+1)
    global Hq += V/4.0 * embed(hilbq, i, sigmaz(SpinBasis(1//2))) * embed(hilbq, mod(i, N)+1, sigmaz(SpinBasis(1//2)))
end

@test Hq.data ≈ NeuralQuantum.to_matrix(H)

@testset "Matrix mapping" begin
    lh = liouvillian(H, [])
    lhq = liouvillian(Hq, [])
    @test NeuralQuantum.to_matrix(lh) ≈ lhq.data

    l0 = liouvillian(ops)
    l0q = liouvillian(DenseOperator(hilbq), opsq)
    @test NeuralQuantum.to_matrix(l0) ≈ l0q.data

    liouv = liouvillian(H, ops)
    liouvq = liouvillian(Hq, opsq)
    @test NeuralQuantum.to_matrix(liouv) ≈ liouvq.data
end
