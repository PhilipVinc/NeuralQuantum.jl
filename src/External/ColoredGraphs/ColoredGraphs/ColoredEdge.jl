using LightGraphs.SimpleGraphs: SimpleEdge, SimpleGraphEdge

import Base: Pair, Tuple, show, ==
using LightGraphs: AbstractEdge, src, dst
using LightGraphs.SimpleGraphs: LightGraphs.SimpleGraphs, SimpleEdge

"""
    ColoredEdge{T<:Integer} <: AbstractColoredEdge

A simple implementation of a colored edge, with a `src` a `dst` and a color
`col` of the edge.
Those properties should be accessed with the methods that work on all
`AbstractColoredEdge` `src`, `dst`, `color`.
"""
struct ColoredEdge{T<:Integer} <: AbstractColoredEdge{T}
    src::T
    dst::T
    col::T
end
ColoredEdge{T}(src,dst,col) where T = ColoredEdge{T}(src,dst,col)
ColoredEdge(t::Tuple)    = ColoredEdge(t[1], t[2], t[3])
ColoredEdge(p::Tuple{Pair,Integer}) = ColoredEdge(p[1].first, p[1].second, p[2])
ColoredEdge(p::Tuple{SimpleEdge,Integer}) = ColoredEdge(src(p[1]), dst(p[1]), p[2])
ColoredEdge(p::SimpleEdge, col) = ColoredEdge(src(p), dst(p), col)
ColoredEdge{T}(p::Tuple{Pair,Integer}) where T<:Integer = ColoredEdge(T(p[1].first), T(p[1].second), T(p[2]))
ColoredEdge{T}(t::Tuple) where T<:Integer = ColoredEdge(T(t[1]), T(t[2]), T(t[3]))
SimpleGraphs.SimpleEdge(ce::AbstractColoredEdge) = SimpleEdge(src(ce), dst(ce))

eltype(e::ET) where ET<:AbstractColoredEdge{T} where T = T

# Accessors
LightGraphs.src(e::ColoredEdge) = e.src
LightGraphs.dst(e::ColoredEdge) = e.dst
color(e::ColoredEdge) = e.col
color(e::SimpleEdge) = 1 # default value when no color

# I/O
show(io::IO, e::AbstractColoredEdge) = print(io, "Colored Edge $(e.src) => $(e.dst) = [$(e.col)]")

# Conversions
Pair(e::AbstractColoredEdge) = Pair(src(e), dst(e))
Tuple(e::AbstractColoredEdge) = (src(e), dst(e), color(e))

ColoredEdge{T}(e::AbstractColoredEdge) where T <: Integer = ColoredEdge{T}(T(e.src), T(e.dst),  T(e.col))

# Convenience functions
Base.reverse(e::T) where T<:AbstractColoredEdge = T(dst(e), src(e), color(e))
==(e1::AbstractColoredEdge, e2::AbstractColoredEdge) = (src(e1) == src(e2) && dst(e1) == dst(e2) && color(e1) == color(e2))
