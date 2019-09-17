(cnet::CachedNet)(σ::LUState) =
    logψ(cnet, σ)

logψ(cnet::CachedNet, σ::LUState{<:DoubleState}) = begin
    σ_r = row(state(σ))
    σ_c = col(state(σ))

    return logψ(cnet.net, cnet.cache, lut(σ),
                config(raw_state(σ_r)), config(raw_state(σ_c)),
                changes(σ_r), changes(σ_c))
end

logψ_and_∇logψ(cnet::CachedNet, σ::LUState) = begin
   # @warn "Inefficient calling logψ_and_∇logψ for cachedNet"
    ∇lnψ = grad_cache(cnet)
    lnψ, ∇lnψ = logψ_and_∇logψ!(∇lnψ, cnet, σ)
    return lnψ, ∇lnψ
end

logψ_and_∇logψ!(∇lnψ, cnet::CachedNet, σ::LUState) = begin
    σ_r = row(state(σ))
    σ_c = col(state(σ))

    lnψ = logψ_and_∇logψ!(∇lnψ, cnet.net, cnet.cache, lut(σ),
                          config(raw_state(σ_r)), config(raw_state(σ_c)),
                          changes(σ_r), changes(σ_c))
    return lnψ, ∇lnψ
end

logψ_Δ(cnet::CachedNet, σ::LUState{<:DoubleState}) = begin
    σ_r = row(state(σ))
    σ_c = col(state(σ))
    return Δ_logψ(cnet.net, cnet.cache, lut(σ),
                  config(raw_state(σ_r)), config(raw_state(σ_c)),
                  changes(σ_r), changes(σ_c))
end

logψ_Δ(cnet::CachedNet, σ::LUState{<:DoubleState}, changes_r, changes_c) = begin
    σ_r = row(state(σ))
    σ_c = col(state(σ))
    return Δ_logψ(cnet.net, cnet.cache, lut(σ),
                  config(raw_state(σ_r)), config(raw_state(σ_c)),
                  changes_r, changes_c)
end

function Δ_logψ_and_∇logψ(cnet::CachedNet, σ::LUState, args...)
    ∇lnψ = grad_cache(cnet)
    return Δ_logψ_and_∇logψ!(∇lnψ, cnet, σ, args...)
end

Δ_logψ_and_∇logψ!(∇lnψ, cnet::CachedNet, σ::LUState{<:DoubleState}) = begin
    σ_r  = row(state(σ))
    σ_c  = col(state(σ))
    Δlnψ = Δ_logψ_and_∇logψ!(∇lnψ,
                  cnet.net, cnet.cache, lut(σ),
                  config(raw_state(σ_r)), config(raw_state(σ_c)),
                  changes(σ_r), changes(σ_c))
    return Δlnψ, ∇lnψ
end

Δ_logψ_and_∇logψ!(∇lnψ, cnet::CachedNet, σ::LUState{<:DoubleState}, changes_r, changes_c) = begin
    σ_r  = row(state(σ))
    σ_c  = col(state(σ))
    Δlnψ = Δ_logψ_and_∇logψ!(∇lnψ,
                  cnet.net, cnet.cache, lut(σ),
                  config(raw_state(σ_r)), config(raw_state(σ_c)),
                  changes_r, changes_c)
    return Δlnψ, ∇lnψ
end

prepare_lut!(σ::LUState, cnet::CachedNet) =
    set_lookup!(lut(σ), cnet.net, cnet.cache, raw_config(state(σ))...)
    #set_lookup!(lut(σ), cnet.net, cnet.cache, config(state(σ))...)

apply_lut_updates!(σ::LUState{<:DoubleState}, cnet::CachedNet) = begin
    σ_r = row(state(σ))
    σ_c = col(state(σ))
    update_lookup!(lut(σ), cnet.net, cnet.cache,
        config(raw_state(σ_r)), config(raw_state(σ_c)),
        changes(σ_r), changes(σ_c))
end

update_lookup!(σ::LUState, cnet::CachedNet) = begin
    # TODO Check if this is correct. I could just rename this, no?
    update_lut!(σ, cnet)
end
