using LightGraphs.SimpleGraphs: SimpleEdge, AbstractSimpleGraph
using LightGraphs.SimpleGraphs: SimpleGraphs, fadj, ne, badj, add_edge!, rem_edge!
using LightGraphs: ne, edgetype, has_edge, is_directed

"""
    AbstractWrappedSimpleGraph{T} <: AbstractSimpleGraph{T}

A wrapper type that wraps a SimpleGraph, used to store additional info in it's
type. Mainly used to store information about the spatial embedding (in 1D,2D)
and directions (x,y,z...) of the underlying graph.

Must implement the field `bare_graph::SG` that should store the underlying
graph.
"""
abstract type AbstractWrappedSimpleGraph{T} <: AbstractSimpleGraph{T} end

## imple
bare(g::AbstractWrappedSimpleGraph) = g.bare_graph

@forward AbstractWrappedSimpleGraph.bare_graph SimpleGraphs.fadj, SimpleGraphs.badj, LightGraphs.ne,
    LightGraphs.edgetype, LightGraphs.is_directed, LightGraphs.has_edge, Base.eltype,
        SimpleGraphs.add_edge!, SimpleGraphs.rem_edge!, LightGraphs.nv, LightGraphs.edges

LightGraphs.ne(g::AbstractWrappedSimpleGraph) = ne(bare(g))
LightGraphs.is_directed(g::Type{<:AbstractWrappedSimpleGraph}) = is_directed(fieldtype(g, :bare_graph))
