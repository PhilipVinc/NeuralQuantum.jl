"""
    PurifiedState{ST<:DualValuedState} <: DualValuedComposedState

State composed by two identical states ST, used to represent states of a
density matrix.
"""
mutable struct PurifiedState{ST<:FiniteBasisState} <: FiniteBasisState
    n_sys::Int # number of states
    n_add::Int # number of states
    σ_sys::ST # left
    σ_add::ST # right

    space_dim::Int
end

## Constructors
PurifiedState(sys::ST, add::ST) where {ST<:FiniteBasisState} =
    PurifiedState(nsites(sys),   nsites(add),
                  deepcopy(sys), deepcopy(add),
                  spacedimension(sys)*spacedimension(add))

PurifiedState{ST}(n, add, i_σ=0) where ST =
    set!(PurifiedState(ST(n, 0), ST(add, 0)), i_σ)

# Property Accessors
@inline spacedimension(state::PurifiedState) = state.space_dim
@inline nsites(state::PurifiedState) = nsites(state.σ_sys) + nsites(state.σ_add)
@inline toint(state::PurifiedState) = _toint(add(state), sys(state)) #toint(state.σ_row, state.σ_col)
@inline index(state::PurifiedState) = index(sys(state))
@inline flipped(a::PurifiedState, b::PurifiedState) = (flipped(a.σ_sys, b.σ_sys), flipped(a.σ_add, b.σ_add))
@inline index_to_int(state::PurifiedState, id) = @error("Undefined: index is on sys, add is undefined.")
@inline index_to_int(state::PurifiedState, id, add::FiniteBasisState) =
        index_to_int(state, id, toint(add))
@inline index_to_int(state::PurifiedState, id, add) = add*spacedimension(sys(state)) + index_to_int(sys(state), id)
@inline eltype(state::PurifiedState) = eltype(sys(state))

# custom accessor
sys(v::PurifiedState) = v.σ_sys
add(v::PurifiedState) = v.σ_add
config(v::PurifiedState) = (config(v.σ_sys), config(v.σ_add))

# checs
same_basis(v::PurifiedState{ST}, v2::PurifiedState{ST2}) where {ST, ST2} =
    ST==ST2 && same_basis(sys(v), sys(v2)) && same_basis(add(v), add(v2))

function flipat!(rng::AbstractRNG, v::PurifiedState, i::Int)
    i > v.n_sys ? flipat!(v.σ_add, i-v.n_sys) : flipat!(v.σ_sys, i)
end

function setat!(v::PurifiedState, i::Int, val)
    i > v.n_sys ? setat!(v.σ_add, i-v.n_sys, val) : setat!(v.σ_sys, i, val)
end

set_index!(v::PurifiedState, i::Integer, a) = set!(v, index_to_int(v, i, a))
function set!(v::PurifiedState, i::Integer)
    i_add, i_sys = divrem(i, spacedimension(v.σ_sys))
    set!(v.σ_sys, i_sys)
    set!(v.σ_add, i_add)

    v
end
set!(v::PurifiedState, sys, add) = set!(sys(v), row) && set!(add(v), col) && v

function rand!(rng::AbstractRNG, state::PurifiedState)
    rand!(rng, state.σ_sys)
    rand!(rng, state.σ_add)
end

set_sys!(state::PurifiedState, i::Integer) = set!(state.σ_sys, i)
set_add!(state::PurifiedState, i::Integer) = set!(state.σ_add, i)

# -------------- Base.show extension for nice printing -------------- #
Base.show(io::IO, mm::MIME"text/plain", bs::PurifiedState) =
    print(io, "PurifiedState($(bs.n_sys), $(bs.n_add)) : \n\tRow: ", bs.σ_sys,"\n\tCol: ", bs.σ_add, "\n")

Base.show(io::IO, bs::PurifiedState) = print(io, "(",bs.σ_sys,", ", bs.σ_add, ")")
