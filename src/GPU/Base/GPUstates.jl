const gpuAState{T} = GPUArray{T,1}
const gpuAStateBatch{T} = GPUArray{T,2}
const gpuAStateBatchVec{T} = GPUArray{T,3}

const gpuADoubleState{T} = NTuple{2, gpuAState{T}} where T
const gpuADoubleStateBatch{T} = NTuple{2, gpuAStateBatch{T}} where T
const gpuADoubleStateBatchVec{T} = NTuple{2, gpuAStateBatchVec{T}} where T

function state_collect(s::Union{gpuADoubleStateBatchVec,gpuADoubleStateBatch,gpuADoubleState})
    return (state_collect(row(s)), state_collect(col(s)))
end

state_collect(s::Union{gpuAStateBatchVec,gpuAStateBatch,gpuAState}) = collect(s)

# efficient state generation for homogeneous spaces on gpu
# could be improved with a custom kernel...
function Random.rand!(rng::AbstractRNG, σ::Union{gpuAState,gpuAStateBatch}, h::HomogeneousFock{N}) where N
    T = eltype(σ)
    rand!(rng, σ) # must find a way to use an RNG in here...
    σ .*= N
    σ .= floor.(σ)
end
