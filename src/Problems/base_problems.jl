abstract type AbstractSteadyStateProblem <: AbstractProblem end
abstract type HermitianMatrixProblem <: AbstractSteadyStateProblem end
abstract type LRhoSquaredProblem <: AbstractSteadyStateProblem end
abstract type OpenTimeEvolutionProblem <: AbstractSteadyStateProblem end
abstract type OperatorEstimationProblem <: AbstractProblem end
