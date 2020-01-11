mutable struct BatchedObsKetSampler{S,V,O,A,R,Pd}
    observables::Dict

    samples::S
    logψ_vals::V
    Oloc_vals::O

    accum::A

    results::R

    parallel_cache::Pd
end

function BatchedObsKetSampler(samples, ψvals, accum, 
                              par_type=automatic_parallel_type())

    Oloc_vals    = similar(ψvals, size(ψvals)[2:end]...)

    par_cache    = parallel_execution_cache(par_type)

    is = BatchedObsKetSampler(Dict(),
                              samples,
                              ψvals,
                              Oloc_vals,
                              accum,
                              Dict(),
                              par_cache)
    return is
end

function add_observable!(is::BatchedObsKetSampler, name::String, obs::AbsLinearOperator)
    is.observables[name] = obs
end

function compute_observables(is::BatchedObsKetSampler)
    for (name, Ô) = is.observables
        is.results[name] = compute_observable(is, Ô)
    end

    return is.results
end

function compute_observable(is::BatchedObsKetSampler, Ô::AbsLinearOperator)
    ch_len         = size(is.Oloc_vals, 2)
    batch_sz       = size(is.Oloc_vals, 1)

    for i=1:ch_len
        for j = 1:batch_sz
            σv = unsafe_get_el(is.samples, j, i)
            init!(is.accum, σv, is.logψ_vals[1,j,i])
            accumulate_connections!(is.accum, Ô, σv)
            O_loc = NeuralQuantum.finalize!(is.accum)
            is.Oloc_vals[j, i] = O_loc
        end
    end

    return stat_analysis(is.Oloc_vals, is.parallel_cache)
end
