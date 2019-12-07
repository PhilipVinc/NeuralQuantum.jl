using NeuralQuantum, Random, Plots

Random.seed!(Random.GLOBAL_RNG, 1234)

# Parameters of the transverse field ising model
N = 20
h = 1.0
J = 1.0

# Constructs the Hilbert space for N 1//2 spins.
hilb = HomogeneousSpin(N, 1//2)

# Builds the hamiltonian
H = LocalOperator(hilb)
for i=1:N
    global H  -= h * sigmax(hilb, i)
    global H  += J * sigmaz(hilb, i) * sigmaz(hilb, mod(i, N)+1)
end

# Build the FFNN with one dense layer and a final sum layer
# instead of using sum, we use sum_autobatch, so that we can correctly
# batch calls to the FFNN.
ch = Chain(Dense(ComplexF64, N, 2*N, af_logcosh), sum_autobatch)

# Wrap the chain (FFNN) into an object correctly describing a Pure State
net = PureStateAnsatz(ch, N)

# Randomize all parameters
init_random_pars!(net, sigma=0.01)

sampl = MetropolisSampler(LocalRule(), 75, N, burn=100)
algo  = SR(ϵ=(0.1), algorithm=sr_cg, precision=1e-3)

is = BatchedSampler(net, sampl, H, algo; batch_sz=16)

optimizer = Optimisers.Descent(0.05)

Evalues = Float64[];
Eerr = Float64[];
for i=1:300
    ldata, prec = sample!(is)
    ob = compute_observables(is)

    println("$i - $ldata")

    push!(Evalues, real(ldata.mean))
    push!(Eerr, ldata.error)
    grad = precondition!(prec, algo, i)
    Optimisers.update!(optimizer, net, grad)
end

plot(Evalues, yerr=Eerr)

# N=20 (thanks netket)
exact = -1.274549484318e00 * 20
hline!([exact, exact])
