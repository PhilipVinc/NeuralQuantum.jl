abstract type SrMat{T} <: LinearMap{T} end

Base.eltype(S::SrMat{T}) where T = T
LinearAlgebra.ishermitian(S::SrMat) = true

# Lazy Algebra: addition/subtraction
add!(S::SrMat, α::Number) = (S.shift += α; return S)
subtract!(S::SrMat, α) = add!(S, -α)
shift!(S::SrMat, α) = (S.shift = α; return S)

#Base.:+(S::SrMat, α::Number) = add!(copy(S), α)
Base.:+(S::SrMat, α::LinearAlgebra.UniformScaling) = add!(copy(S), α.λ)

# Constructor
SrMatrix(T, O::AbstractMatrix, α::Number=0) = begin
    if T <: Real
        S = SrMatrixR(O, α)
    else
        S = SrMatrixC(O, α)
    end
    return S
end

mutable struct SrMatrixC{T,Tv<:AbstractMatrix{T},Ct,Cr} <: SrMat{T}
    O::Tv
    shift::T

    # cache
    ṽ::Ct
    res::Cr
end

SrMatrixC(O::AbstractMatrix, α::Number=0) =
    SrMatrixC(O, convert(eltype(O), α),
              similar(O, size(O,2)), similar(O, size(O,1)))

Base.copy(S::SrMatrixC) = SrMatrixC(S.O, S.shift,
                                    S.ṽ, S.res)

Base.size(S::SrMatrixC) = (size(S.O,1), size(S.O,1))
Base.isreal(S::SrMatrixC) = isreal(S.O)
LinearAlgebra.issymmetric(S::SrMatrixC) = isreal(S)

function init!(S::SrMatrixC, O)
    # Check size
    if size(S.O) != size(O)
        S.O = conj(O)
        S.ṽ = similar(O, size(O,2))
    else
        S.O .= conj.(O)
    end

    return S
end

function LinearAlgebra.mul!(C::AbstractVector, S::SrMatrixC{T}, F::AbstractVector, α::Number=one(T), β::Number=zero(T)) where {T}
    O     = S.O
    shift = S.shift

    #cache
    ṽ   = S.ṽ
    res = S.res

    # calculation
    mul!(ṽ, O', F)
    mul!(res, O, ṽ)
    N = size(O,2)

    if β == 0
        C .= α .* (F .* shift .+ res ./ N)
    else
        C .= α .* (F .* shift .+ res ./ N) .+ β .* C
    end
end



##
mutable struct SrMatrixR{T,Tv<:AbstractMatrix{T},Ct,Cr} <: SrMat{T}
    Oᵣ::Tv
    Oᵢ::Tv
    shift::T

    # cache
    ṽ::Ct
    res::Cr
end

SrMatrixR(O::AbstractMatrix, α::Number=0) =
    SrMatrixR(real(O),
              imag(O),
              convert(eltype(O), α),
              similar(O, real(eltype(O)), size(O,2)),
              similar(O, size(O,1)))

Base.copy(S::SrMatrixR) = SrMatrixR(S.Oᵣ, S.Oᵢ, S.shift,
                                    S.ṽ, S.res)

Base.size(S::SrMatrixR) = (size(S.Oᵣ,1), size(S.Oᵣ,1))

init!(S::SrMatrixR, O) = begin
    # Check size
    # Check size
    if size(S.Oᵣ) != size(O)
        S.Oᵣ = real(O)
        S.Oᵢ = imag(O)
        S.ṽ  = similar(O, real(eltype(O)), size(O,2))
    else
        S.Oᵣ .= real.(O)
        S.Oᵢ .= imag.(O)
    end

    return S
end

function LinearAlgebra.mul!(C::AbstractVector, S::SrMatrixR{T}, F::AbstractVector, α::Number=one(T), β::Number=zero(T)) where {T}
    Oᵣ     = S.Oᵣ
    Oᵢ     = S.Oᵢ
    shift = S.shift

    #cache
    ṽ   = S.ṽ
    res = S.res

    # calculation real part
    mul!(ṽ, Oᵣ', F)
    mul!(res, Oᵣ, ṽ)

    # calculation imaginary part
    mul!(ṽ, Oᵢ', F)
    mul!(res, Oᵢ, ṽ, one(eltype(ṽ)), one(eltype(ṽ)))

    N = size(Oᵣ,2)

    if β == 0
        C .= α .* (F .* shift .+ res ./ N)
    else
        C .= α .* (F .* shift .+ res ./ N) .+ β .* C
    end
end
