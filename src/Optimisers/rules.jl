# Code taken from FluxML/Optimisers.jl

"""
    Descent(η)

Classic gradient descent optimiser with learning rate `η`.
For each parameter `p` and its gradient `δp`, this runs `p -= η*δp`.
"""
mutable struct Descent
  eta::Float64
end

function apply(o::Descent, x, x̄, state = nothing)
  x̄ .* o.eta, state
end

struct Identity end

function apply(o::Identity, x, x̄, state = nothing)
  if x === x̄
    return 0, state
  else
    return x .+ x̄, state
  end
  x̄, state
end

mutable struct Nesterov
  lr::AbstractFloat
  μ::AbstractFloat
  gclip::AbstractFloat
end

function apply(o::Nesterov, x, Δ, v=zero(x)) # v = old speed
  lr = o.lr
  μ  = o.μ

  d = @. μ^2 * v - (1+μ) * lr * Δ
  @. v = μ*v - lr*Δ

  -d, state
end


"""
    ∇clip!(⁠∇W, clip_val)

clips the norm of the vector `∇W` to clip_val if it is bigger than that,
otherwise leaves `∇W` unchanged
"""
function ∇clip!(∇W::AbstractArray, clip_val)
    if clip_val == 0
        ∇W
    else
        ∇norm = norm(∇W)
        if ∇norm <= clip_val
            return ∇W
        else
            return lmul!(clip_val/∇norm, ∇W)
        end
    end
end
