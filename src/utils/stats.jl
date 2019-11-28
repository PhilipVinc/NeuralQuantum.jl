struct Measurement
    mean
    error

    variance
    tau
    R
end

function stat_analysis(vals::AbstractMatrix)
    tmp = similar(vals, size(vals, 1))
    n    = size(vals, 1)
    L    = size(vals, 2)

    tmp = similar(vals, size(vals, 1))

    μ_chains = mean!(tmp, vals)
    μ        = mean(μ_chains)

    var_chains = var(vals, dims=2, mean=μ_chains)
    var_μ_ch   = var(μ_chains)
    var_μ      = var(vals, mean=μ)


    μ_err = sqrt(var_μ_ch/n)
    μ_var = mean(var_chains)

    t = var_μ_ch/var_μ
    corr = max(0.0, 0.5 * ( t * L - 1))
    R = sqrt((L-1)/L + t)

    return Measurement(μ,
            μ_err, μ_var, corr, R)
end


Base.show(io::IO, ::MIME"text/plain", v::Measurement) =
    print(io, _meas_to_str(v))

Base.show(io::IO, v::Measurement) =
    print(io, _meas_to_str(v))

function _meas_to_str(v)
    μ = v.mean
    if μ isa Complex
        sgn = sign(imag(μ)) == 1 ? "+" : "-"
        μs = @sprintf "%6.4f %s %6.4f im" real(μ) sgn abs(imag(μ))
    else
        μs = @sprintf "%6.4f" real(μ)
    end

    @sprintf "(%s) ± %6.4f [var=%6.4f, tau=%6.4f, R=%6.4f]" μs v.error v.variance v.tau v.R
end
