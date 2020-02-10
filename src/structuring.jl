export destructure

function _restructure(m, xs)
  i = 0
  fmap(m) do x
    x isa AbstractArray || return x
    x = reshape(xs[i.+(1:length(x))], size(x))
    i += length(x)
    return x
  end
end

"""
    destructure(m)
Flatten a model's parameters into a single weight vector.
    julia> m = Chain(Dense(10, 5, σ), Dense(5, 2), softmax)
    Chain(Dense(10, 5, σ), Dense(5, 2), softmax)
    julia> θ, re = destructure(m);
    julia> θ
    67-element Array{Float32,1}:
    -0.1407104
    ...
The second return value `re` allows you to reconstruct the original network after making
modifications to the weight vector (for example, with a hypernetwork).
    julia> re(θ .* 2)
    Chain(Dense(10, 5, σ), Dense(5, 2), softmax)
"""
function destructure(m)
  xs = Zygote.Buffer([])
  fmap(m) do x
    x isa AbstractArray && push!(xs, x)
    return x
  end
  return vcat(vec.(copy(xs))...), p -> _restructure(m, p)
end
