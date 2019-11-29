using Pkg;
Pkg.activate(joinpath(@__DIR__, "..")); Pkg.instantiate()
Pkg.activate(); Pkg.instantiate()

pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Documenter, NeuralQuantum

makedocs(
    modules   = [NeuralQuantum],
    format    = Documenter.HTML(prettyurls = haskey(ENV, "CI")),
    sitename  = "NeuralQuantum.jl",
    authors   = "Filippo Vicentini",
    pages     = [
            "Home"          => "index.md",
            "Manual"        => Any[
                "Basics"        => "basics.md",
                "Problems"      => "problems.md",
                "Algorithms"    => "algorithms.md",
                "Networks"      => "networks.md",
                "Optimizers"    => "optimizers.md"
            ],
            "Internals"     => Any[
                "States"        => "states.md",
            ]
    ]
)

deploydocs(
    repo   = "github.com/PhilipVinc/NeuralQuantum.jl.git",
    target = "build"
    )
