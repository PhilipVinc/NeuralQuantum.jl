mutable struct LUState{S<:State, L} <: FiniteBasisState
    state::S
    lookup::L
    valid::Bool
end

LUState(s, lu) = LUState(s, lu, false)
export LUState

# accessors
state(s::LUState) = s.state
lut(s::LUState)   = s.lookup
isvalid(s::LUState) = s.valid

raw_state(s::LUState) = raw_state(state(s))
invalidate!(s::LUState) = s.valid = false
validate!(s::LUState) = s.valid = true
# Modifiers
init_lut!(s::LUState, net::NeuralNetwork, force::Bool=false) = begin
    if isvalid(s) && !force
        return update_lut!(s, net)
    end
    apply_chages!(state(s))
    prepare_lut!(s, net)
    validate!(s)
    return s
end

init_lut!(s::State, net) = nothing
update_lut!(s::State, net) = nothing

update_lut!(s::LUState, net::NeuralNetwork) = begin
    apply_lut_updates!(s, net)
    apply_chages!(state(s))
    return s
end

clear_changes!(s::LUState) = clear_changes!(state(s))

## State Interface : custom accessors
@inline config(s::LUState) = (invalidate!(s); config(state(s)) )

## State Interface : Property Accesors
@inline spacedimension(s::LUState) = spacedimension(state(s))
@inline nsites(s::LUState) = nsites(state(s))
@inline local_dimension(s::LUState) = local_dimension(state(s))
@inline eltype(s::LUState) = eltype(state(s))

toint(s::LUState) = (invalidate!(s); toint(state(s)))
index(s::LUState) = (invalidate!(s); return index(state(s)))
index_to_int(s::LUState) = (invalidate!(s); index_to_int(state(s)))

## State Interface : Checks
same_basis(s1::LUState, s2::LUState) =
    same_basis(state(s1), state(s2))

## State Interface : flip operations
# This is the standard method, that applies all exhisting transformations
function flipat!(rng::AbstractRNG, s::LUState, args...)
    invalidate!(s)
    return flipat!(rng, state(s), args...)
end

function flipat_fast!(rng::AbstractRNG, s::LUState, args...)
    return flipat_fast!(rng, state(s), args...)
end

setat!(s::LUState, args...) = (invalidate!(s);
    setat!(state(s), args...); s)

set_index!(s::LUState, val) = (invalidate!(s);
    set_index!(state(s), val); s)

set!(s::LUState, val) = begin
    invalidate!(s)
    set!(state(s), val)
    s
end
add!(s::LUState, val) = (invalidate!(s);
    add!(state(state), val); s)
rand!(rng, s::LUState) = (invalidate!(s);
    rand!(rng, state(s)); s)

Base.show(io::IO, ::MIME"text/plain", s::LUState) =
    print(io, "LUState: $(state(s)) valid: $(s.valid)")
