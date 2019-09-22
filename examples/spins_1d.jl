# Load dependencies
using NeuralQuantum, QuantumLattices
using Logging, Printf, ValueHistories

# Select the numerical precision
T      = Float64
# Select how many sites you want
sites  = [3, 3]
Nsites = prod(sites)

# Create the lattice as [Nx, Ny, Nz]
lattice = SquareLattice(sites, PBC=true)
# Create the hamiltonian for the QI model
Ĥ = quantum_ising_ham(lattice, g=1.0, V=2.0)
# Create the Problem (cost function) for the given hamiltonian
# targeting the ground state.

#-- Observables
# Define the local observables to look at.
Sx  = QuantumLattices.LocalObservable(Ĥ, sigmax, Nsites)
Sy  = QuantumLattices.LocalObservable(Ĥ, sigmay, Nsites)
Sz  = QuantumLattices.LocalObservable(Ĥ, sigmaz, Nsites)
# Create the problem object with all the observables to be computed.
oprob = ObservablesProblem(Sx, Sy, Sz, Ĥ)


# Define the Neural Network. A RBM with N visible spins and α=2
net  = RBM(Complex{T}, Nsites, 1)
# Create a cached version of the neural network for improved performance.
cnet = cached(net)
# Chose a sampler. Options are FullSumSampler() which sums over the whole space
# ExactSampler() which does exact sampling or MCMCSamler which does a markov
# chain.
sampl = MCMCSampler(Metropolis(), 1000, burn=50)
# Chose a sampler for the observables.
osampl = FullSumSampler()

# Chose the gradient descent algorithm (alternative: Gradient())
# for more information on options type ?SR
algo  = SR(ϵ=T(0.001), use_iterative=true)
# Optimizer: how big the steps of the descent should be
optimizer = Optimisers.Descent(0.005)

# Create a multithreaded Iterative Sampler.
is = MTIterativeSampler(cnet, sampl, prob, algo)
ois = MTIterativeSampler(cnet, osampl, oprob)

# Create the structure to store all output data
minimization_data = MVHistory()
Δw = grad_cache(cnet)

# Solve iteratively the problem

# Solve iteratively the problem
for i=1:500
    # Sample the gradient
    grad_data  = sample!(is)
    obs_data = sample!(ois)

    # Logging
    @printf "%4i -> %+2.8f %+2.2fi --\t \t-- %+2.5f\n" i real(grad_data.L) imag(grad_data.L) real(obs_data.ObsAve[1])
    push!(minimization_data, :loss, grad_data.L)
    for (name,val)=zip(obs_data.ObsNames, obs_data.ObsAve)
        push!(minimization_data, Symbol(name), val)
    end

    succ = precondition!(Δw.tuple_all_weights, algo , grad_data, i)
    !succ && break
    Optimisers.update!(optimizer, cnet, Δw)
end

# Optional: compute the exact solution
en, st = eigenstates(DenseOperator(ham))
E_gs = real(minimum(en))
ψgs = first(st)
ESx = real(expect(SparseOperator(Sx), ψgs))
ESy = real(expect(SparseOperator(Sy), ψgs))
ESz = real(expect(SparseOperator(Sz), ψgs))
exacts = Dict("Sx"=>ESx, "Sy"=>ESy, "Sz"=>ESz)
## - end Optional

using Plots
data = minimization_data 

iter_cost, cost = get(minimization_data[:loss])
pl1 = plot(iter_cost, real(cost))
hline!(pl1, [E_gs, E_gs])

iter_mx, mx = get(minimization_data[:obs_1])
pl2 = plot(iter_mx, real(mx))
hline!(pl2, [ESx,ESx])

plot(pl1, pl2, layout=(2,1))
