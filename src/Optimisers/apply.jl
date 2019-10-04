# Code taken from FluxML/Optimisers.jl

const Param{T<:Number} = Union{AbstractArray{T},T}

_apply(opt, x, x̄, state) = apply(opt, x, x̄, state)
_apply(opt, x, x̄, ::Nothing) = apply(opt, x, x̄)

# Immutable updates
update!(opt, x, xb, state=nothing) = _update!(opt, x, xb, state)
update(opt, x, xb, state=nothing)  = _update(opt, x, xb, state)

function _update(opt, x::Param, x̄::Param, state = nothing)
  Δ, state = _apply(opt, x, x̄, state)
  return x .- Δ, state
end

# Mutable updates

# Figure out if we can do in-place
inplace(x, y) = false
inplace(x, y::Nothing) = true
inplace(x::AbstractArray, x̄::AbstractArray) = true
inplace(x, x̄::NamedTuple) = all(inplace(getfield(x, f), getfield(x̄, f)) for f in fieldnames(typeof(x̄)))

function _update!(opt, x::AbstractArray{<:Number}, x̄::AbstractArray, state = nothing)
  Δ, state = _apply(opt, x, x̄, state)
  #println("fr = ", x)
  #println("to = ", Δ)
  x .-= Δ
  #println("after = ", x)
  return state
end

#function _update!(opt, x::Tuple, x̄::Tuple, state=nothing)
function _update!(opt, x, x̄::Tuple, state=nothing)
  for f in propertynames(x̄)
    f̄ = getindex(x̄, f)

    f̄ === nothing || _update!(opt, getindex(x, f), f̄, state)
  end
end

function _update!(opt, x, x̄::NamedTuple, state=nothing)
  for f in propertynames(x̄)
    f == :tuple_all_weights && continue
    f̄ = getproperty(x̄, f)

    f̄ === nothing || _update!(opt, getfield(x, f), f̄, state)
  end
end

function _update!(opt, x, x̄, state=nothing)
  for f in propertynames(x̄)
    if f isa Int
      f̄ = getindex(x̄, f)
      f̄ === nothing || _update!(opt, getindex(x, f), f̄, state)
    else
      f̄ = getproperty(x̄, f)
      f̄ === nothing || _update!(opt, getfield(x, f), f̄, state)
    end
  end
end
