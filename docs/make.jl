using Documenter, NeuralQuantum

makedocs(
    modules   = [NeuralQuantum],
    format    = Documenter.HTML(),
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
    target = "build",
    deps   = nothing,
    make   = nothing
)
