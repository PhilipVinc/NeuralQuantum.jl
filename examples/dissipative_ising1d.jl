using NeuralQuantum, QuantumOpticsBase, ProgressMeter
using NeuralQuantum: unsafe_get_el

N = 7
g = 0.4
V = 2.0

hilb = HomogeneousSpin(N,1//2)
hilb = HomogeneousFock(N,2)

ops = []
H = LocalOperator(hilb)

Sx = LocalOperator(hilb)
Sy = LocalOperator(hilb)
Sz = LocalOperator(hilb)

for i=1:N
    global H += g/2.0 * sigmax(hilb, i)
    global H += V/4.0 * sigmaz(hilb, i) * sigmaz(hilb, mod(i, N)+1)

    global Sx += sigmax(hilb, i)/N
    global Sy += sigmay(hilb, i)/N
    global Sz += sigmaz(hilb, i)/N

    push!(ops, sigmam(hilb, i))
end

liouv = liouvillian(H, ops)

sampl = MetropolisSampler(LocalRule(), 125, N, burn=100)
#sampl = ExactSampler(5000)
algo  = SR(ϵ=(0.001), algorithm=sr_cholesky)
#algo  = Gradient()

net  = NDM(Float64, N, 1, 1, af_sigmoid)
is = BatchedSampler(net, sampl, liouv, algo; batch_sz=16)
add_observable(is, "Sx", Sx)
add_observable(is, "Sy", Sy)
add_observable(is, "Sz", Sz)

optimizer = Optimisers.Descent(0.01)

Evalues = Float64[];
Eerr = Float64[];
for i=1:200
    ldata, prec = sample!(is)
    ob = compute_observables(is)

    println("$i - $ldata")

    push!(Evalues, ldata.mean)
    push!(Eerr, ldata.error)
    grad = precondition!(prec, algo, i)
    Optimisers.update!(optimizer, net, grad)
end

plot(Evalues, yerr=Eerr, yscale=:log10)
