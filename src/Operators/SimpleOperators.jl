"""
    sigmax(hilbert::AbstractHilbert, site::Int)

Builds a σₓ operator acting on the `site`-th site of the Many body Hilbert
space `hilbert`.

Note: M-dimensional Hilbert spaces are treated as spin (M-1)//2 spins.
"""
function QuantumOpticsBase.sigmax(h::AbstractHilbert, i::Int)
    N = shape(h)[i]
    S = (N-1)//2 # map Fock spaces to Spins

    # Build the local operator matrix
    D   = ComplexF64[complex(sqrt(real((S + 1)*2*a - a*(a+1)))) for a=1:N-1]
    mat = diagm(1=>D, -1=>D)

    return KLocalOperatorRow(h, [i], mat)
end

"""
    sigmay(hilbert::AbstractHilbert, site::Int)

Builds a σ_y operator acting on the `site`-th site of the Many body Hilbert
space `hilbert`.

Note: Hilbert must have local space dimension 2 on that site.
"""
function QuantumOpticsBase.sigmay(h::AbstractHilbert, i::Int)
    N = shape(h)[i]
    S = (N-1)//2 # map Fock spaces to Spins

    # Build the local operator matrix
    D   = ComplexF64[1im*complex(sqrt(real((S + 1)*2*a - a*(a+1)))) for a=1:N-1]
    mat = diagm(-1=>D, 1=>-D)

    return KLocalOperatorRow(h, [i], mat)
end

"""
    sigmaz(hilbert::AbstractHilbert, site::Int)

Builds a σ_z operator acting on the `site`-th site of the Many body Hilbert
space `hilbert`.

Note: M-dimensional Hilbert spaces are treated as spin (M-1)//2 spins.
"""
function QuantumOpticsBase.sigmaz(h::AbstractHilbert, i::Int)
    N = shape(h)[i]
    S = (N-1)//2 # map Fock spaces to Spins

    # Build the local operator matrix
    D   = ComplexF64[complex(2*m) for m=S:-1:-S]
    mat = diagm(0 => D)

    return KLocalOperatorRow(h, [i], mat)
end

"""
    sigmam(hilbert::AbstractHilbert, site::Int)

Builds a σ₋ operator acting on the `site`-th site of the Many body Hilbert
space `hilbert`.

Note: M-dimensional Hilbert spaces are treated as spin (M-1)//2 spins.
"""
function QuantumOpticsBase.sigmam(h::AbstractHilbert, i::Int)
    N = shape(h)[i]
    S = (N-1)//2 # map Fock spaces to Spins

    S2 = (S+1)*S
    D = [complex(sqrt(float(S2 - m*(m-1)))) for m=S:-1:-S+1]
    mat = diagm(-1 => D)

    return KLocalOperatorRow(h, [i], mat)
end

"""
    sigmap(hilbert::AbstractHilbert, site::Int)

Builds a σ₊ operator acting on the `site`-th site of the Many body Hilbert
space `hilbert`.

Note: M-dimensional Hilbert spaces are treated as spin (M-1)//2 spins.
"""
function QuantumOpticsBase.sigmap(h::AbstractHilbert, i::Int)
    N = shape(h)[i]
    S = (N-1)//2 # map Fock spaces to Spins

    S2 = (S+1)*S
    D = ComplexF64[complex(sqrt(float(S2 - m*(m+1)))) for m=S-1:-1:-S]
    mat = diagm(1 => D)

    return KLocalOperatorRow(h, [i], mat)
end

"""
    destroy(hilbert::AbstractHilbert, site::Int)

Builds a bosonic destruction operator acting on the `site`-th site of the Many
body Hilbert space `hilbert`.

Note: spin-`m//2` spaces are treated as fock spaces with dimension `m+1`
"""
function QuantumOpticsBase.destroy(h::AbstractHilbert, i::Int)
    N = shape(h)[i] - 1

    D = complex.(sqrt.(1.:N))
    mat = diagm(1 => D)

    return KLocalOperatorRow(h, [i], mat)
end

"""
    create(hilbert::AbstractHilbert, site::Int)

Builds a bosonic creation operator acting on the `site`-th site of the Many
body Hilbert space `hilbert`.

Note: spin-`m//2` spaces are treated as fock spaces with dimension `m+1`
"""
function QuantumOpticsBase.create(h::AbstractHilbert, i::Int)
    N = shape(h)[i] - 1

    D = complex.(sqrt.(1.:N))
    mat = diagm(-1 => D)

    return KLocalOperatorRow(h, [i], mat)
end
