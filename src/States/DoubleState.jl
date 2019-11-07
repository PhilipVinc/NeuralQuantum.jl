"""
    DoubleState{ST<:DualValuedState} <: DualValuedComposedState

State composed by two identical states ST, used to represent states of a
density matrix.
"""
mutable struct DoubleState{ST<:FiniteBasisState} <: FiniteBasisState
    n::Int64 # number of sites
    σ_row::ST # left
    σ_col::ST # right

    space_dim::Int
end

## Constructors
DoubleState(sr::FiniteBasisState, sc::FiniteBasisState=deepcopy(sr)) =
    same_basis(sr, sc) ? DoubleState(nsites(sr),
                                     deepcopy(sr),
                                     deepcopy(sc),
                                     spacedimension(sr)^2) : error("not same basis")

# Specialized Constructors
DoubleState{ST}(n, i_σ=0) where ST = set!(DoubleState(ST(n, 0), ST(n, 0)), i_σ)


# Property Accessors
@inline spacedimension(state::DoubleState) = state.space_dim
@inline nsites(state::DoubleState) = 2*nsites(state.σ_row)
@inline local_dimension(state::DoubleState{ST}) where {ST} = local_dimension(ST)
@inline Base.eltype(state::DoubleState) = eltype(row(state))

toint(state::DoubleState) = _toint(col(state), row(state)) #toint(state.σ_row, state.σ_col)
index(state::DoubleState) = toint(state) + 1 # was before
index_to_int(state::DoubleState, id) = (id -1)
flipped(a::DoubleState, b::DoubleState) = (flipped(a.σ_row, b.σ_row), flipped(a.σ_col, b.σ_col))

# custom accessor
row(v::DoubleState) = v.σ_row
col(v::DoubleState) = v.σ_col
@inline config(v::DoubleState) = (config(v.σ_row), config(v.σ_col))
#@inline config(v::DoubleState) = return config(v.σ_row), config(v.σ_col)

# checks
same_basis(v::DoubleState{ST}, v2::DoubleState{ST2}) where {ST, ST2} =
    ST==ST2 && same_basis(row(v), row(v2)) && same_basis(col(v), col(v2))

function flipat!(rng::AbstractRNG, v::DoubleState, i::Int)
    i > v.n ? flipat!(rng, v.σ_row, i-v.n) : flipat!(rng, v.σ_col, i)
end

function flipat_fast!(rng::AbstractRNG, v::DoubleState, i::Int)
    i > v.n ? flipat_fast!(rng, v.σ_row, i-v.n) : flipat_fast!(rng, v.σ_col, i)
end

function setat!(v::DoubleState, i::Int, val)
    i > v.n ? setat!(v.σ_row, i-v.n, val) : setat!(v.σ_col, i, val)
end

"""
    apply!(state::State, changes)

Applies the changes `changes` to the `state`.
If `state isa DoubleState` then single-value changes
are applied to the columns of the state (in order to
compute matrix-operator products). Otherwise it should
be a tuple with changes of row and columns
"""
function apply!(state::DoubleState, changes::StateChanges)
    for (site, val) = changes
        #setat!(col(state), site, val)
        # The code below automatically applies it only
        # to columns.
        setat!(state, site, val)
    end
end

#function apply!(state::DoubleState, changes_r::StateChanges, changes_c::StateChanges)
function apply!(state::DoubleState, changes_r, changes_c)
    apply!(row(state), changes_r)
    apply!(col(state), changes_c)
    return state
end

set_index!(v::DoubleState, i::Integer) = set!(v, index_to_int(v, i))
function set!(v::DoubleState, i::Integer)
    row = div(i, spacedimension(v.σ_row)) #row = i>>(nsites(state.σ_row))
    col = i - row*spacedimension(v.σ_row) #col = i - (row<< nsites(state.σ_row))

    set!(v.σ_col, row) #i
    set!(v.σ_row, col) #j
    return v
end
set!(v::DoubleState, i_row, i_col) = (set!(row(v), i_row); set!(col(v), i_col);  v)

function Random.rand!(rng::AbstractRNG, state::DoubleState)
    rand!(rng, state.σ_row)
    rand!(rng, state.σ_col)
    state
end

# -------------- Base.show extension for nice printing -------------- #
Base.show(io::IO, mm::MIME"text/plain", bs::DoubleState) =
    print(io, "DoubleState(",bs.n,") : \n\tRow: ", bs.σ_row,"\n\tCol: ", bs.σ_col)

Base.show(io::IO, bs::DoubleState) = print(io, "(",bs.σ_row,", ", bs.σ_col, ")")
