using NNlib: kernel_size,  output_size, channels_in, channels_out
using NNlib: ∇conv_data!, ∇conv_filter!, conv!

function QMaskedConv(filter_sz::NTuple{N,Integer},
                     channels=1=>1;
                     bias_first=false) where N
    padding = Int[]
    for f=filter_sz
        p = bias_first ? f : f-1
        push!(padding, p)
        push!(padding, 0)
    end
    padding = Tuple(padding)

    return Conv(filter_sz, channels, logℒ, pad=padding)
end

struct ConvCache{Tc,To,Td,Tb,Cd,Co,Cfo,Cdo,Te}
    σ::Tc
    out::To
    δℒℒ::Td

    b_rshpd::Tb
    cdims::Cd
    conv_out::Co
    conv_filter_out::Cfo
    conv_data_out::Cdo

    out2::Te
    valid::Bool
end

cache(l::Conv, in_T, in_sz) = begin
    if length(in_sz) == length(size(l.weight))-2
        in_sz   = (in_sz...,1,1)
        @info "did"
    end
    in_rspd = similar(l.weight, in_T, in_sz)

    b_rshpd = reshape(l.bias, map(_->1, l.stride)..., :, 1)
    cdims = DenseConvDims(in_sz, size(l.weight);
                          stride=l.stride, padding=l.pad,
                          dilation=l.dilation)
    xT = in_T
    wT = eltype(l.weight)
    conv_T = promote_type(xT, wT)
    conv_out = similar(l.weight, conv_T,
                    NNlib.output_size(cdims)...,
                    NNlib.channels_out(cdims), in_sz[end])

    out = similar(conv_out, promote_type(conv_T, eltype(l.bias)))

    δℒℒ = similar(out)
    #
    conv_filter_out = similar(δℒℒ, kernel_size(cdims)...,
                        channels_in(cdims), channels_out(cdims))
    conv_data_out   = similar(δℒℒ, NNlib.input_size(cdims)...,
                        channels_in(cdims),
                            size(δℒℒ)[end])

    ConvCache(in_rspd, out, δℒℒ,
              b_rshpd, cdims, conv_out, conv_filter_out, conv_data_out,
              0, false)
end

layer_out_type_size(l::Conv, in_T, in_sz) = begin
    if length(in_sz) == length(size(l.weight))-2
        in_sz   = (in_sz...,1,1)
        @info "did"
    end
    cdims = DenseConvDims(in_sz, size(l.weight);
                          stride=l.stride, padding=l.pad,
                          dilation=l.dilation)
    conv_T = promote_type(in_T, eltype(l.weight))
    return promote_type(conv_T, eltype(l.bias)), (NNlib.output_size(cdims)..., NNlib.channels_out(cdims)..., in_sz[end])
end

function (l::Conv)(c::ConvCache, x)
    σ    = l.σ
    b    = c.b_rshpd
    cdims = c.cdims

    x = isnothing(c.σ) ? x : copyto!(c.σ, x)

    conv!(c.conv_out, x, l.weight, cdims)
    c.conv_out .+= b
    c.out .= σ.(c.conv_out)
    return c.out
end

function backprop(∇, l::Conv, c::ConvCache, δℒ)
    δℒℒ = c.δℒℒ

    # Compute the actual sensitivity
    copyto!(δℒℒ, δℒ)
    if l.σ == identity
        δℒℒ .= c.conv_out
    elseif l.σ == logℒ
        δℒℒ .*= ∂logℒ.(c.conv_out)
    else
        throw("notknown $(l.σ)")
    end

    conv_filter_out = ∇conv_filter!(c.conv_filter_out, c.σ, δℒℒ, c.cdims)
    conv_data_out   = ∇conv_data!(c.conv_data_out, δℒℒ, l.weight, c.cdims)

    ∇.weight  .= c.conv_filter_out
    ∇.bias    .= sum(δℒℒ) # sum(δℒℒ, dims=1)
    return c.conv_data_out
end
