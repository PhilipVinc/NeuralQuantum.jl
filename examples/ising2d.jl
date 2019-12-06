using NeuralQuantum, Random, Plots

Random.seed!(Random.GLOBAL_RNG, 1234)

# Parameters of the transverse field ising model
dims = [5,5]
h = 3.0
J = 1.0

# Constructs the Hilbert space for N 1//2 spins.
N    = prod(dims)
hilb = HomogeneousSpin(N, 1//2)

coord(i,j) = i + (j-1)*dims[2]

# Builds the hamiltonian
H = LocalOperator(hilb)
for i=1:dims[1]
    for j=1:dims[2]
        p = coord(i,j)
        global H  -= h * sigmax(hilb, p)
        hop = J * sigmaz(hilb, p) * sigmaz(hilb, coord(mod(i, dims[1])+1,j)) +
                J * sigmaz(hilb, p) * sigmaz(hilb, coord(i, mod(j, dims[2])+1))
        global H += hop
    end
end

net  = RBM(Float32, N, 1, af_logcosh)
init_random_pars!(net, sigma=0.01)

sampl = MetropolisSampler(LocalRule(), 75, N, burn=100)
algo  = SR(ϵ=(0.1), algorithm=sr_qlp)

is = BatchedSampler(net, sampl, H, algo;
                    batch_sz=16)

optimizer = Optimisers.Descent(0.1)

Evalues = Float64[];
Eerr = Float64[];
for i=1:1000
    ldata, prec = sample!(is)
    ob = compute_observables(is)

    println("$i - $ldata")

    push!(Evalues, real(ldata.mean))
    push!(Eerr, ldata.error)
    grad = precondition!(prec, algo, i)
    Optimisers.update!(optimizer, net, grad)
end

plot(Evalues, yerr=Eerr)

exact = -80.13310152422413
hline!([exact, exact])
