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
valid(s::LUState) = s.valid

raw_state(s::LUState) = raw_state(state(s))

# Modifiers
init_lut!(s::LUState, net::NeuralNetwork) = begin
    apply_chages!(state(s))
    prepare_lut!(s, net)
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
@inline config(s::LUState) = config(state(s))

## State Interface : Property Accesors
@inline spacedimension(s::LUState) = spacedimension(state(s))
@inline nsites(s::LUState) = nsites(state(s))
@inline local_dimension(s::LUState) = local_dimension(state(s))
@inline eltype(s::LUState) = eltype(state(s))

toint(s::LUState) = toint(state(s))
index(s::LUState) = (s.valid = false; return index(state(s)))
index_to_int(s::LUState) = (s.valid = false; index_to_int(state(s)))

## State Interface : Checks
same_basis(s1::LUState, s2::LUState) =
    same_basis(state(s1), state(s2))

## State Interface : flip operations
# This is the standard method, that applies all exhisting transformations
flipat!(rng::AbstractRNG, s::LUState, args...) = (s.valid = false;
    flipat!(rng, state(s), args...))

flipat_fast!(rng::AbstractRNG, s::LUState, args...) = (s.valid = true;
    flipat_fast!(rng, state(s), args...))

setat!(s::LUState, args...) = (s.valid = false;
    setat!(state(s), args...))

set_index!(s::LUState, val) = (s.valid = false;
    set_index!(state(s), val))
set!(s::LUState, val) = (s.valid = false;
    set!(state(s), val))
add!(s::LUState, val) = (s.valid = false;
    add!(state(state), val))
rand!(rng, s::LUState) = (s.valid = false;
    rand!(rng, state(s)))

Base.show(io::IO, ::MIME"text/plain", s::LUState) =
    print(io, "LUState: $(state(s)) valid: $(s.valid)")
