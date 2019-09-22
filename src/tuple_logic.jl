export weight_tuple

add_vector_field(net::CachedNet, grad) = add_vector_field(net, grad)
add_vector_field(net, grad) = last(_vectorize_gradient(net, grad))


#TODO I can plug here to customize this for every network type,
# ie by passing a dict of fieldnames...
_vectorize_gradient(net, x::NamedTuple) =
    _vectorize_gradient(x, zeros(out_type(net), compute_len(x)))
function _vectorize_gradient(x::NamedTuple, vec::Vector, start=1)
    i = 0
    d=Dict{Symbol, Any}()
    for f=fieldnames(typeof(x))
        di, val = _vectorize_gradient(getfield(x,f), vec, start+i)
        i += di; push!(d, f=>val)
    end
    start == 1 && push!(d, :tuple_all_weights=>[vec])
    #@views start != 1 && push!(d, :tuple_all_weights=>vec[start:start+i-1])
    i, (;d...)
end

function _vectorize_gradient(x::Tuple, vec::Vector, start)
    i = 0
    d=Vector()
    for f=fieldnames(typeof(x))
        di, val = vectorize_tuple(getfield(x,f), vec, start+i)
        i += di; push!(d, val)
    end
    i, Tuple(d)
end

function _vectorize_gradient(x::AbstractArray{<:Number}, vec::Vector, start)
    @views data_vec = vec[start:start+length(x)-1]
    reshpd_params = reshape(data_vec, size(x))
    reshpd_params .= x
    return length(x), reshpd_params
end


#=
function compute_len(vals::Union{NamedTuple,Tuple})
    tot = 0
    for f=fieldnames(typeof(vals))
        tot += compute_len(getfield(vals, f))
    end
    tot
end =#
"""
    compute_len(vals::Union{NamedTuple, Tuple})

Computes the total length of all the arrays and matrices present in this
structure and it's children.
"""
compute_len(vals::Union{NamedTuple, Tuple}) =
    mapreduce(f->compute_len(getfield(vals, f)), +, fieldnames(typeof(vals)))
compute_len(vals::AbstractArray) = mapreduce(compute_len, +, vals)
compute_len(vals::AbstractArray{<:Number}) = length(vals)

derivative_tuple(cnet::CachedNet) = derivative_tuple(cnet.net)
derivative_tuple(net) = weight_tuple(net, fieldnames(typeof(net)),
                                     Vector{out_type(net)}())[2]


"""
    weight_tuple(net)

Returns a named tuple holding all the fields in the network, and an
extra field named `tuple_all_weights` who has the same type
"""
weight_tuple(cnet::CachedNet) = weight_tuple(cnet.net)
weight_tuple(net) = weight_tuple(net, fieldnames(typeof(net)))[2]
function weight_tuple(x, fnames, vec=Vector{weight_type(x)}(), start=1)
    i = 0
    d=Dict{Symbol, Any}()
    for f=fnames
        di, val = weight_tuple(getfield(x,f), vec, start+i)
        i += di; push!(d, f=>val)
    end
    #start == 1 && push!(d, :tuple_all_weights=>[vec])
    i, (;d...)
end

function weight_tuple(x::Tuple, vec::Vector, start)
    i = 0
    d=Vector()
    for f=fieldnames(typeof(x))
        di, val = weight_tuple(getfield(x,f), vec, start+i)
        i += di; push!(d, val)
    end
    i, Tuple(d)
end

function weight_tuple(x::AbstractArray{<:Number}, vec::Vector, start)
    length(vec) < start+length(x)-1 && resize!(vec, start+length(x)-1)
    @views data_vec = vec[start:start+length(x)-1]
    reshpd_params = reshape(data_vec, size(x))
    reshpd_params .= x
    return length(x), reshpd_params
end

##
function batched_weight_tuple(grad_tup, bsz=1)
    all_weights = grad_tup.tuple_all_weights
    all_weghts_new = [similar(w, length(w), bsz) for w=all_weights][1]
    return batched_weight_tuple(grad_tup, fieldnames(typeof(grad_tup)), all_weghts_new)[2]
end

function batched_weight_tuple(x, fnames, vec, start=1)
    i = 0
    d=Dict{Symbol, Any}()
    for f=fnames
        f==:tuple_all_weights && continue
        di, val = batched_weight_tuple(getfield(x,f), vec, start+i)
        i += di; push!(d, f=>val)
    end
    start == 1 && push!(d, :tuple_all_weights=>[vec])
    i, (;d...)
end

function batched_weight_tuple(x::Tuple, vec::Matrix, start)
    i = 0
    d=Vector()
    for f=fieldnames(typeof(x))
        di, val = batched_weight_tuple(getfield(x,f), vec, start+i)
        i += di; push!(d, val)
    end
    i, Tuple(d)
end

function batched_weight_tuple(x::AbstractArray{<:Number}, vec::Matrix, start)
    @assert length(vec) >= start+length(x)-1
    bsz = size(vec, 2)

    @views data_vec = vec[start:start+length(x)-1, :]
    reshpd_params = reshape(data_vec, size(x)..., bsz)
    reshpd_params .= x
    if reshpd_params isa Base.ReshapedArray
        reshpd_params = StridedView(reshpd_params)
    end
    return length(x), reshpd_params
end # ? stridedView?
