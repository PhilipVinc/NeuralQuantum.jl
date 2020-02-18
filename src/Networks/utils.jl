# Arrays

glorot_uniform(dims...) = glorot_uniform(STD_REAL_PREC, dims...)
glorot_normal(dims...) = glorot_normal(STD_REAL_PREC, dims...)
rescaled_normal(scale::Real, dims...) = rescaled_normal(STD_REAL_PREC, scale, dims...)

glorot_uniform(T::Type, dims...) = (rand(T, dims...) .- T(0.5)) .* sqrt(T(24)/sum(dims))
glorot_normal(T::Type, dims...) = randn(T, dims...) .* sqrt(T(2)/sum(dims))
rescaled_normal(T::Type, scale::Real, dims::Integer...) = randn(T, dims...) .* T(scale) .* sqrt(T(24)/sum(dims))
