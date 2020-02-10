abstract type AbstractObservableAccumulator <: AbstractAccumulator end
abstract type AbstractMachineAccumulator <: AbstractAccumulator end

abstract type AbstractMachineValAccumulator <: AbstractMachineAccumulator end
abstract type AbstractMachineGradAccumulator <: AbstractMachineAccumulator end

init!(acc::AbstractObservableAccumulator, σ, ψ_σ, ∇ψ_σ) = init!(acc, σ, ψ_σ)
