struct ModifiedState{S<:State, C<:AbstractStateChanges} <: FiniteBasisState
    state::S
    changes::C
end

ModifiedState(state::State) = ModifiedState(state, StateChanges(state))

# accessors
raw_state(s::ModifiedState) = s.state
changes(s::ModifiedState)   = s.changes

# Modifiers
apply_changes!(s::ModifiedState) = begin
    apply!(raw_state(s), changes(s))
    zero!(changes(s))
    return s
end
apply_warn_raw!(s::ModifiedState) = begin
    #@warn "applying changes not optimized"
    apply_changes!(s)
    return raw_state(s)
end
clear_changes!(s::ModifiedState) = zero!(changes(s))

## For DoubleState
apply_chages!(s::DoubleState) = nothing
function apply_chages!(s::DoubleState{<:ModifiedState})
    apply_changes!(row(s))
    apply_changes!(col(s))
    return s
end
clear_changes!(s::DoubleState) = nothing
function clear_changes!(s::DoubleState{<:ModifiedState})
    clear_changes!(row(s))
    clear_changes!(col(s))
    return s
end

raw_config(s::DoubleState) = config(s)
raw_config(s::DoubleState{<:ModifiedState}) = (raw_config(s.σ_row), raw_config(s.σ_col))
## end



## State Interface : custom accessors
@inline config(s::ModifiedState) = config(apply_warn_raw!(s))
@inline raw_config(s::ModifiedState) = config(raw_state(s))

## State Interface : Property Accesors
@inline spacedimension(s::ModifiedState) = spacedimension(raw_state(s))
@inline nsites(s::ModifiedState) = nsites(raw_state(s))
@inline local_dimension(s::ModifiedState{S,C}) where {S,C} = local_dimension(S)
@inline local_dimension(s::Type{ModifiedState{S,C}}) where {S,C} = local_dimension(S)
@inline eltype(s::ModifiedState) = eltype(raw_state(s))

toint(s::ModifiedState) = toint(apply_warn_raw!(s))
index(s::ModifiedState) = index(apply_warn_raw!(s))
index_to_int(s::ModifiedState) = index_to_int(apply_warn_raw!(s))

## State Interface : Checks
same_basis(s1::FiniteBasisState, s2::FiniteBasisState) =
    nsites(s1)==nsites(s2) && local_dimension(s1) == local_dimension(s2) && eltype(s1) == eltype(s2)

## State Interface : Operations on tuples
_toint(left::ModifiedState, right::ModifiedState) = _toint(apply_warn_raw!(left), apply_warn_raw!(right))

## State Interface : flip operations
# This is the standard method, that applies all exhisting transformations
function flipat!(rng::AbstractRNG, state::ModifiedState, i::Int)
    old_val = config(state)[i]
    return _flipat!(rng, state, i, old_val)
end

# This is a faster method, that assumes that you are flipping spins that
# are not flipped in the changes
function flipat_fast!(rng::AbstractRNG, state::ModifiedState, i::Int)
    old_val = config(raw_state(state))[i]
    return _flipat!(rng, state, i, old_val)
end

# Inner method actually performing the flip
function _flipat!(rng::AbstractRNG, state::ModifiedState, i::Int, old_val)
    # Generate new value
    N = local_dimension(raw_state(state))
    T = eltype(raw_state(state))
    new_val = T(rand(rng, 0:(N-1)))
    while new_val == old_val
        new_val = T(rand(rng, 0:(N-1)))
    end

    # Modify
    push!(changes(state), (i, new_val))
    return (old_val, new_val)
end

function setat!(state::ModifiedState, i::Int, val)
    old_val = config(raw_state(state))[i]
    push!(changes(state), (i, val))

    return old_val
end

function setat_fast!(state::ModifiedState, i::Int, val)
    old_val = config(raw_state(state))[i]
    push!(changes(state), (i, val))

    return old_val
end


set_index!(s::ModifiedState, val) = (zero!(changes(s)); set_index!(raw_state(s), val))
set!(s::ModifiedState, val) = (zero!(changes(s)); set!(raw_state(s), val); return s)
add!(s::ModifiedState, val) = add!(apply_warn_raw!(state), val)
rand!(rng, s::ModifiedState) = (zero!(changes(s));
    rand!(rng, raw_state(s)))

Base.show(io::IO, ::MIME"text/plain", s::ModifiedState) =
    print(io, "ModifiedState: $(raw_state(s)) with changes $(changes(s))")
