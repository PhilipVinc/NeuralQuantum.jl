using Pkg;
Pkg.activate(joinpath(@__DIR__, "..")); Pkg.instantiate()
Pkg.activate(); Pkg.instantiate()

pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Documenter, NeuralQuantum

makedocs(
    modules   = [NeuralQuantum],
    sitename  = "NeuralQuantum.jl",
    authors   = "Filippo Vicentini",
    pages     = [
            "Home"          => "index.md",
            "Manual"        => Any[
                "Basics"        => "basics.md",
                "Liouvillian"   => "liouvillian.md",
                "Networks"      => "networks.md",
                "Samplers"      => "samplers.md",
                "SR"            => "algorithms.md",
                "Optimizers"    => "optimizers.md"
            ],
            "Reference" => "reference.md"
    ],
    format    = Documenter.HTML(
                    prettyurls = haskey(ENV, "CI"),
                    mathengine = MathJax())
)

deploydocs(
    repo   = "github.com/PhilipVinc/NeuralQuantum.jl.git",
    target = "build"
    )
