using NeuralQuantum, QuantumOpticsBase, Statistics
using Test

N = 4
g = 0.7
V = 2.0

hilb = HomogeneousSpin(N)

H = LocalOperator(hilb)
for i=1:N
    global H += g/2.0 * sigmax(hilb, i)
    global H  += V/4.0 * sigmaz(hilb, i) * sigmaz(hilb, mod(i, N)+1)
end

Sx = LocalOperator(hilb)
Sy = LocalOperator(hilb)
Sz = LocalOperator(hilb)
for i=1:N
    global Sx += sigmax(hilb, i)/N
    global Sy += sigmay(hilb, i)/N
    global Sz += sigmaz(hilb, i)/N
end

prob  = NeuralQuantum.HamiltonianGSEnergyProblem(H)

net  = RBM( N, 2)

sampl = MetropolisSampler(LocalRule(), 300, N, burn=30)
#sampl = ExactSampler(1000)
algo  = SR(ϵ=(0.001), use_iterative=true)
algo  = Gradient()

is = BatchedSampler(net, sampl, prob, algo; batch_sz=32)
res, grad = sample!(is)

# exact value
Hmat = Matrix(H)
psi  = ket(net, hilb)

E_ex = psi'*Hmat*psi / (psi'psi)

# precise sampling
sampl_ex = ExactSampler(1000)
algo     = SR()
is = BatchedSampler(net, sampl_ex, prob, algo; batch_sz=32)

res, grad = sample!(is)

@test abs(res.mean - E_ex) < res.error

#
