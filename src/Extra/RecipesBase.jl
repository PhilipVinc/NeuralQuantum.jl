using RecipesBase

@recipe function f(y::AbstractArray{<:NeuralQuantum.Measurement})
    yerror := uncertainty.(y)
    value.(y)
end

@recipe function f(x::AbstractArray, y::AbstractArray{<:NeuralQuantum.Measurement})
    yerror := uncertainty.(y)
    x, value.(y)
end

@recipe function f(x::AbstractArray{<:NeuralQuantum.Measurement}, y::AbstractArray)
    xerror := uncertainty.(x)
    value(x), y
end
