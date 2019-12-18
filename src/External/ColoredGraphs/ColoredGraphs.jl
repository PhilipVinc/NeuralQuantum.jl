module ColoredGraphs

using LightGraphs
using MacroTools: @forward

import Base:
    eltype, show, ==, Pair, Tuple, copy, length, issubset, reverse, zero, in, iterate

import LightGraphs:
    _NI, AbstractGraph, AbstractEdge, AbstractEdgeIter,
    src, dst, edgetype, nv, ne, vertices, edges, is_directed,
    has_vertex, has_edge, inneighbors, outneighbors,

    indegree, outdegree, degree, has_self_loops, num_self_loops, insorted

export ColoredGraph, ColoredDiGraph, ColoredEdge

export HyperCube, translational_symm_table

"""
    AbstractColoredEdge{T} <: AbstractEdge{T}

Abstract base type for an edge with a color.
"""
abstract type AbstractColoredEdge{T} <: AbstractEdge{T} end

"""
    AbstractColoredGraph{T} <: AbstractGraph{T}

Abstract base type for colored graphs.
"""
abstract type AbstractColoredGraph{T} <: AbstractGraph{T} end

include("ColoredGraphs/ColoredEdge.jl")
include("ColoredGraphs/ColoredEdgeIter.jl")
include("ColoredGraphs/ColoredGraph.jl")

include("WrappedSimpleGraphs/AbstractWrappedSimpleGraph.jl")
include("WrappedSimpleGraphs/HyperCube.jl")

end # module
