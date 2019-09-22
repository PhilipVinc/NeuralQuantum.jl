using NeuralQuantum, Test
using NeuralQuantum: lookup, init_lut!, flipat_fast!, update_lut!, logψ_Δ, lut, Δ_logψ_and_∇logψ!
num_types = [Float32, Float64]

re_machines = Dict()
im_machines = Dict()

ma = (T, N) -> NDM(T, N, 2, 3)
re_machines["NDM"] = ma

ma = (T, N) -> RBMSplit(T, N, 2)
im_machines["RBMSplit"] = ma

all_machines = merge(re_machines, im_machines)

N = 4
T = last(num_types)

@testset "Test LUT: Update_LUT $name" for name=keys(all_machines)
    for T=num_types
        net = all_machines[name](T,N)
        isnothing(lookup(net)) && continue

        cnet = cached(net)

        v = state_lut(T, SpinBasis(1//2)^N, net)
        zero!(v)
        init_lut!(v, cnet, true)
        # compute exact
        results = [];
        for i=1:spacedimension(v)
            set_index!(v, i)
            init_lut!(v, cnet, true)

            for j=unique(rand(1:nsites(v), nsites(v)))
                old, new = flipat_fast!(v, j)
            end
            update_lut!(v, cnet)
            u_lut = deepcopy(lut(v))
            init_lut!(v, cnet, true)
            i_lut = lut(v)
            good = Bool[]
            for f=propertynames(i_lut)
                push!(good, getproperty(u_lut, f) ≈ getproperty(i_lut, f))
            end
            push!(results, all(good))
        end
        @test all(results)
    end
end

@testset "Test LUT: Logψ_Δ $name" for name=keys(all_machines)
    for T=num_types
        net = all_machines[name](T,N)
        isnothing(lookup(net)) && continue

        cnet = cached(net)

        vr = state(T, SpinBasis(1//2)^N, net)
        zero!(vr)

        v = state_lut(T, SpinBasis(1//2)^N, net)
        zero!(v)
        init_lut!(v, cnet)
        # compute exact
        Δ_vals = []; exact_vals = [];
        eval_vals = []; exact = [];
        for i=1:spacedimension(v)
            set_index!(vr, i)
            set_index!(v, i)
            init_lut!(v, cnet, true)

            lnψ_old = cnet(vr)
            for j=unique(rand(1:nsites(v), nsites(v)))
                old, new = flipat_fast!(v, j)
                setat!(vr, j, new)
            end
            lnψ_new  = cnet(vr)
            lnψ_Δ_ex = lnψ_new-lnψ_old
            lnψ_Δ    = logψ_Δ(cnet, v)
            push!(Δ_vals, lnψ_Δ)
            push!(exact_vals, lnψ_new-lnψ_old)
            push!(eval_vals, logψ(cnet, v))
            push!(exact, lnψ_new)
        end
        @test Δ_vals ≈ exact_vals
        @test eval_vals ≈ exact
    end
end


@testset "Test LUT: Cached Gradient $name" for name=keys(merge(re_machines, im_machines))
    for T=num_types
        net = all_machines[name](T,N)
        isnothing(lookup(net)) && continue

        cnet = cached(net)
        cder = grad_cache(net)

        vr = state(T, SpinBasis(1//2)^N, net)
        zero!(vr)

        v = state_lut(T, SpinBasis(1//2)^N, net)
        zero!(v)
        init_lut!(v, cnet)

        # compute exact
        grads = [];
        for i=1:spacedimension(v)
            set_index!(vr, i)
            set_index!(v, i)
            init_lut!(v, cnet)

            for j=1:nsites(v)
                old, new = flipat_fast!(v, j)
                setat!(vr, j, new)
            end

            der_c = ∇logψ(cnet, vr)
            der_l = ∇logψ(cnet, v)
            for f=propertynames(der_l)
                ∇  = getproperty(der_c,  f)
                c∇ = getproperty(der_l, f)
                push!(grads, ∇ ≈ c∇)
            end
        end
        @test all(grads)
    end
end

@testset "Test LUT: Cached Gradient $name" for name=keys(merge(re_machines, im_machines))
    for T=num_types
        net = all_machines[name](T,N)
        isnothing(lookup(net)) && continue

        cnet = cached(net)
        cder = grad_cache(net)

        vr = state(T, SpinBasis(1//2)^N, net)
        zero!(vr)

        v = state_lut(T, SpinBasis(1//2)^N, net)
        zero!(v)
        init_lut!(v, cnet)

        # compute exact
        grads_ex   = Bool[]
        grads_diff = Bool[]
        diffs      = Bool[]
        for i=1:spacedimension(v)
            set_index!(vr, i)
            set_index!(v, i)
            init_lut!(v, cnet)

            for j=1:nsites(v)
                old, new = flipat_fast!(v, j)
                setat!(vr, j, new)
            end

            der_c = ∇logψ(cnet, vr)
            der_l = ∇logψ(cnet, v)
            diffval_l = logψ_Δ(cnet, v)
            der_a = grad_cache(cnet)
            diffval, der_a = Δ_logψ_and_∇logψ!(der_a, cnet, v)
            push!(diffs, diffval_l ≈ diffval )
            for f=propertynames(der_l)
                ∇  = getproperty(der_c,  f)
                c∇ = getproperty(der_l, f)
                a∇ = getproperty(der_a, f)
                push!(grads_ex, ∇ ≈ c∇)
                push!(grads_diff, a∇ ≈  c∇)
            end
        end
        @test all(grads_ex)
        @test all(grads_diff)
        @test all(diffs)
    end
end
