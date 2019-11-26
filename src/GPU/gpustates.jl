const gpuAState{T} = GPUArray{T,1}
const gpuAStateBatch{T} = GPUArray{T,2}
const gpuAStateBatchVec{T} = GPUArray{T,3}

const gpuADoubleState{T} = NTuple{2, gpuAState{T}} where T
const gpuADoubleStateBatch{T} = NTuple{2, gpuAStateBatch{T}} where T
const gpuADoubleStateBatchVec{T} = NTuple{2, gpuAStateBatchVec{T}} where T

function statecollect(s::Union{gpuADoubleStateBatchVec,gpuADoubleStateBatch,gpuADoubleState})
    return (statecollect(row(s)), statecollect(col(s)))
end

function statecollect(s::Union{gpuAStateBatchVec,gpuAStateBatch,gpuAState})
    return collect(s)
end

# efficient state generation for homogeneous spaces on gpu
# could be improved with a custom kernel...
function Random.rand!(rng::AbstractRNG, σ::Union{gpuAState,gpuAStateBatch}, h::HomogeneousHilbert{N}) where N
    T = eltype(σ)
    rand!(rng, σ) # must find a way to use an RNG in here...
    σ .*= N
    σ .= floor.(σ)
end
