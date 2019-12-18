using LightGraphs.SimpleGraphs: SimpleEdge, AbstractSimpleGraph
using LightGraphs.SimpleGraphs: SimpleGraphs

export color, colors

mutable struct ColoredGraph{SGT<:AbstractSimpleGraph,T} <: AbstractColoredGraph{T}
    uncolored_graph::SGT
    colmap::Dict{SimpleEdge{T}, Vector{T}}
end

function ColoredGraph(g::AbstractSimpleGraph)
    T=eltype(g)
    cmap = Dict{SimpleEdge{T}, Vector{T}}()
    for e=edges(g)
        cmap[e] = [1]
        if !is_directed(g)
            cmap[reverse(e)] = [1]
        end
    end
    return ColoredGraph(g, cmap)
end

ColoredGraph(n::T=0) where T = ColoredGraph(SimpleGraph(n), Dict{SimpleEdge{T}, Vector{T}}())
ColoredDiGraph(n::T=0) where T = ColoredGraph(SimpleDiGraph(n), Dict{SimpleEdge{T}, Vector{T}}())

Base.eltype(x::ColoredGraph{SGT,T}) where {SGT,T} = T
LightGraphs.edgetype(x::ColoredGraph{SGT,T}) where {SGT,T}  = ColoredEdge{T}

LightGraphs.nv(g::AbstractColoredGraph) = nv(g.uncolored_graph)
LightGraphs.ne(g::AbstractColoredGraph) = begin
    tot = sum(length.(values(colmap(g))))
    is_directed(g) && return tot
    return Int(tot/2)
end
LightGraphs.vertices(g::AbstractColoredGraph) = vertices(g.uncolored_graph)
LightGraphs.edges(g::AbstractColoredGraph) = ColoredEdgeIter(g)

colmap(g::AbstractColoredGraph) = g.colmap
uncolored(g::AbstractColoredGraph) = g.uncolored_graph

set_colors!(g::AbstractColoredGraph, src, dst, cols::AbstractVector) = begin
    colmap(g)[SimpleEdge(src,dst)] = cols
    !is_directed(g) ? colmap(g)[SimpleEdge(dst, src)] = cols : nothing;
    return nothing
end

add_color!(g::AbstractColoredGraph, src, dst, col) = begin
    edge = SimpleEdge(src,dst)

    if edge ∉ keys(colmap(g))
        colmap(g)[edge] = Int[]
    end
    edge_cols = colmap(g)[edge]
    col ∈ edge_cols && return false
    push!(edge_cols, col)

    # If is not directed add the reverse connection
    if !is_directed(g)
        edge = SimpleEdge(dst,src)
        if edge ∉ keys(colmap(g))
            colmap(g)[edge] = Int[]
        end
        edge_cols = colmap(g)[edge]
        push!(edge_cols, col)
    end
    return true
end

colors(g::AbstractColoredGraph, src, dst) = colors(g, SimpleEdge(src,dst))
colors(g::AbstractColoredGraph, e) = colmap(g)[e]
colored(g::AbstractColoredGraph, e::SimpleEdge, col_id) = ColoredEdge(e, colors(g, e)[col_id])
colored(g::AbstractColoredGraph, src, dst, col)         = ColoredEdge(src, dst, colors(g, src, dst)[col])

LightGraphs.has_vertex(g::AbstractColoredGraph, v::Integer) = v in vertices(g)

# vertex properties
LightGraphs.inneighbors(g::AbstractColoredGraph, v::Integer) = inneighbors(uncolored(g),v)
LightGraphs.outneighbors(g::AbstractColoredGraph, v::Integer) = outneighbors(uncolored(g),v)

# specific implementations
#LightGraphs.is_directed(g::ColoredGraph) = is_directed(uncolored(g))
LightGraphs.is_directed(::Type{<:ColoredGraph{SGT,T}}) where {SGT, T}= is_directed(SGT)
LightGraphs.has_edge(g::ColoredGraph, s, d) = has_edge(uncolored(g), s, d)

function LightGraphs.has_edge(g::ColoredGraph, s, d, c)
    has_edge(uncolored(g), s, d) || return false
    return c ∈ colors(g, s, d)
end
LightGraphs.has_edge(g::ColoredGraph, e::ColoredEdge) = has_edge(g, src(e), dst(e), color(e))

# modifications
SimpleGraphs.add_edge!(g::AbstractColoredGraph, e::AbstractColoredEdge) =
    add_edge!(g, src(e), dst(e), color(e))
SimpleGraphs.add_edge!(g::AbstractColoredGraph, src, dst, col) = begin
    add_edge!(uncolored(g), src, dst)
    return add_color!(g, src, dst, col)
end

SimpleGraphs.rem_edge!(g::AbstractColoredGraph, e) = rem_edge!(g, src(e), dst(e))
SimpleGraphs.rem_edge!(g::AbstractColoredGraph, src, dst) = begin
    !rem_edge!(uncolored(g))  && return false
    delete!(colmap(g), SimpleEdge(src,dst))
    !is_directed(g) && delete!(colmap(g), SimpleEdge(dst, src))
    return true
end

SimpleGraphs.add_vertex!(g::AbstractColoredGraph) = add_vertex!(uncolored(g))
SimpleGraphs.add_vertices!(g::AbstractColoredGraph, n) = add_vertices!(uncolored(g), n)

SimpleGraphs.rem_vertex!(g::AbstractColoredGraph, v) = begin
    dsts = outneighbors(g, v)
    !rem_vertex!(uncolored(g)) && return false
    for dst=dsts
        delete!(colmap(g), SimpleEdge(v,dst))
    end
end

# I/O
show(io::IO, g::ColoredGraph) = print(io, "Colored Graph with $(nv(g)) vertices, $(ne(g)) edges")
