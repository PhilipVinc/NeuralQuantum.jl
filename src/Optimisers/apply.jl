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

function _update!(opt, x::Tuple, x̄::Tuple, state=nothing)
  for f in propertynames(x̄)
    f̄ = getfield(x̄, f)
#  println(f,"-->",f̄)
    f̄ === nothing || _update!(opt, getfield(x, f), f̄, state)
  end
end

function _update!(opt, x, x̄::NamedTuple, state=nothing)
  for f in propertynames(x̄)
    f == :tuple_all_weights && continue
    f̄ = getproperty(x̄, f)
#    println(f,"-->",f̄)
    # println(f,"-->", typeof(getfield(x, f)))

    f̄ === nothing || _update!(opt, getfield(x, f), f̄, state)
  end
end

function _update!(opt, x, x̄, state=nothing)
  for f in propertynames(x̄)
    f̄ = getproperty(x̄, f)
#    println(f,"-->",f̄)
    # println(f,"-->", typeof(getfield(x, f)))

    f̄ === nothing || _update!(opt, getfield(x, f), f̄, state)
  end
end

# Package Integration

using Requires

@init @require Colors="5ae59095-9a9b-59fe-a467-6f913c188581" begin
  function _update(opt, x::Colors.RGB{T}, x̄::NamedTuple) where T
    Colors.RGB{T}(clamp(_update(opt, x.r, x̄.r)[1], 0, 1),
                  clamp(_update(opt, x.g, x̄.g)[1], 0, 1),
                  clamp(_update(opt, x.b, x̄.b)[1], 0, 1)), nothing
  end
end
