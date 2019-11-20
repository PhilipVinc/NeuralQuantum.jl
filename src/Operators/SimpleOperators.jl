function QuantumOpticsBase.sigmax(h::AbstractBasis, i::Int)
    @assert shape(h)[i] == 2
    mat = Matrix{ComplexF64}([[0.0 1.0]; [1.0 0.0]])
    return KLocalOperatorRow(h, [i], mat)
end

function QuantumOpticsBase.sigmay(h::AbstractBasis, i::Int)
    @assert shape(h)[i] == 2
    mat = Matrix{ComplexF64}([[0.0 -1.0im]; [1.0im 0.0]])
    return KLocalOperatorRow(h, [i], mat)
end

function QuantumOpticsBase.sigmaz(h::AbstractBasis, i::Int)
    @assert shape(h)[i] == 2
    mat = Matrix{ComplexF64}([[1.0 0.0]; [0.0 -1.0]])
    return KLocalOperatorRow(h, [i], mat)
end

function QuantumOpticsBase.sigmam(h::AbstractBasis, i::Int)
    @assert shape(h)[i] == 2
    mat = Matrix{ComplexF64}([[0.0 0.0]; [1.0 0.0]])
    return KLocalOperatorRow(h, [i], mat)
end

function QuantumOpticsBase.sigmap(h::AbstractBasis, i::Int)
    @assert shape(h)[i] == 2
    mat = Matrix{ComplexF64}([[0.0 1.0]; [0.0 0.0]])
    return KLocalOperatorRow(h, [i], mat)
end
