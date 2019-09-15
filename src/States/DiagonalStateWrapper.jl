struct DiagonalStateWrapper{T} <: FiniteBasisState where {T<:DoubleState}
    parent::T
end

# accessor
Base.parent(σ::DiagonalStateWrapper) = σ.parent

# Property Accessors
spacedimension(s::DiagonalStateWrapper) = spacedimension(row(s.parent))
nsites(s::DiagonalStateWrapper) = nsites(row(s.parent))
toint(s::DiagonalStateWrapper) = toint(row(s.parent))
index(s::DiagonalStateWrapper) = index(row(s.parent))
index_to_int(s::DiagonalStateWrapper, id) = index_to_int(row(s.parent), id)
flipped(a::DiagonalStateWrapper, b::DiagonalStateWrapper) =
    flipped(a.parent, b.parent)
@inline eltype(state::DiagonalStateWrapper) = eltype(state.parent)
@inline config(state::DiagonalStateWrapper) = config(state.parent)

zero!(s::DiagonalStateWrapper) = zero!(s.parent)
base_state(s::DiagonalStateWrapper) = s.parent

# Operations
function flipat!(rng::AbstractRNG, s::DiagonalStateWrapper, i)
    old, new = flipat!(rng, row(s.parent), i)
    setat!(col(s.parent), i, new)
    return (old, new)
end

function setat!(s::DiagonalStateWrapper, i, val)
    setat!(row(s.parent), i, val)
    setat!(col(s.parent), i, val)
    return s
end

set_index!(v::DiagonalStateWrapper, i::Integer) = set!(v, index_to_int(v, i))
function set!(s::DiagonalStateWrapper, i)
    set!(row(s.parent), i)
    set!(col(s.parent), i)
    return s
end

function add!(s::DiagonalStateWrapper, i)
    add!(row(s.parent), i)
    add!(col(s.parent), i)
    return s
end

function rand!(rng::AbstractRNG, s::DiagonalStateWrapper)
    rand!(rng, row(s.parent))
    set!(col(s.parent), toint(row(s.parent)))
    return s
end

# -------------- Base.show extension for nice printing -------------- #
Base.show(io::IO, mm::MIME"text/plain", bs::DiagonalStateWrapper) =
    print(io, "Diagonal State: $(bs.parent)\n")

Base.show(io::IO, bs::DiagonalStateWrapper) = print(io, "DiagSt: ", bs.parent)
