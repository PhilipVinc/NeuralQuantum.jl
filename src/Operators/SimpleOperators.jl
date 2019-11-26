function QuantumOpticsBase.sigmax(h::AbstractHilbert, i::Int)
    @assert shape(h)[i] == 2
    mat = Matrix{ComplexF64}([[0.0 1.0]; [1.0 0.0]])
    return KLocalOperatorRow(h, [i], mat)
end

function QuantumOpticsBase.sigmay(h::AbstractHilbert, i::Int)
    @assert shape(h)[i] == 2
    mat = Matrix{ComplexF64}([[0.0 -1.0im]; [1.0im 0.0]])
    return KLocalOperatorRow(h, [i], mat)
end

function QuantumOpticsBase.sigmaz(h::AbstractHilbert, i::Int)
    @assert shape(h)[i] == 2
    mat = Matrix{ComplexF64}([[1.0 0.0]; [0.0 -1.0]])
    return KLocalOperatorRow(h, [i], mat)
end

function QuantumOpticsBase.sigmam(h::AbstractHilbert, i::Int)
    @assert shape(h)[i] == 2
    mat = Matrix{ComplexF64}([[0.0 0.0]; [1.0 0.0]])
    return KLocalOperatorRow(h, [i], mat)
end

function QuantumOpticsBase.sigmap(h::AbstractHilbert, i::Int)
    @assert shape(h)[i] == 2
    mat = Matrix{ComplexF64}([[0.0 1.0]; [0.0 0.0]])
    return KLocalOperatorRow(h, [i], mat)
end

function QuantumOpticsBase.destroy(h::AbstractHilbert, i::Int)
    n   = shape(h)[i] - 1
    mat = diagm(ComplexF64, 1=>sqrt.(1:n))
    return KLocalOperatorRow(h, [i], mat)
end

function QuantumOpticsBase.create(h::AbstractHilbert, i::Int)
    n   = shape(h)[i] - 1
    mat = diagm(ComplexF64, -1=>sqrt.(1:n))
    return KLocalOperatorRow(h, [i], mat)
end
