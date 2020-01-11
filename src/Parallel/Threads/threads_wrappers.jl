export @MT

struct MTBatchSampler{A}
    samplers::A
end

macro MT_call(fun, val)
    push!(fun.args, Expr(:kw, :par_type, val))
    return quote
        $(esc(fun))
    end
end

macro MT(fun)
    push!(fun.args, Expr(:kw, :par_type, :_tc))
    return quote
        # a global thread_cache
        #TODO make it gensymmed
        global _tc = ThreadsCache()
        s = Vector{Any}(undef, Threads.nthreads())
        Threads.@threads for i=1:Threads.nthreads()
            s[Threads.threadid()] = $(esc(fun))
        end

        # make it type stable
        T = typeof(first(s))
        sT = Vector{T}(undef, length(s))
        sT .= s
        MTBatchSampler(sT)
    end
end

function sample!(s::MTBatchSampler)
    data_1 = Vector{Any}()
    data_prec = Vector{Any}(undef, Threads.nthreads())
    Threads.@threads for i=1:Threads.nthreads()
        L, prec = sample!(s.samplers[Threads.threadid()])
        Threads.threadid() == 1 && push!(data_1, L)
        data_prec[Threads.threadid()] = prec
    end
    T = typeof(first(data_prec))
    data_prec = Vector{T}(data_prec)
    return data_1[1], data_prec
end

function compute_observables!(s::MTBatchSampler)
    data = Vector{Any}()
    Threads.@threads for i=1:Threads.nthreads()
        res = compute_observables!(s.samplers[i])
        Threads.threadid() == 1 && push!(data, res)
    end
    return data[1]
end

function add_observable!(s::MTBatchSampler, name, obs)
    Threads.@threads for i=1:Threads.nthreads()
        add_observable!(s.samplers[i], name, obs)
    end
    return s
end

function precondition!(data::Vector, params, iter_n)
    parallel_preconditioner = false
    data1 = first(data)
    if params isa SR
        if first(data1.S) isa SrMat
            if first(data1.S).parallel_cache isa ThreadsCache
                parallel_preconditioner = true
            end
        end
    end

    if parallel_preconditioner
        _par_precondition!(data, params, iter_n)
    else
        precondition!(data1, params, iter_n)
    end
end

function _par_precondition!(data, params::SR, iter_n)
    res_data = Vector{Any}()
    Threads.@threads for i=1:Threads.nthreads()
        res = _precondition!(data[Threads.threadid()], params, iter_n)
        Threads.threadid() == 1 && push!(res_data, res)
    end
    return res_data[1]
end
