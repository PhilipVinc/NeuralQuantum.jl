GraphPlot._src_index(e::LightGraphs.AbstractEdge, g::LightGraphs.AbstractGraph) = LightGraphs.src(e)
GraphPlot._dst_index(e::LightGraphs.AbstractEdge, g::LightGraphs.AbstractGraph) = LightGraphs.dst(e)

function cgplot(g::AbstractColoredGraph; kw...)
    if :edgelabel ∉ keys(kw)
        clist = []
        for e=edges(g)
            push!(clist, color(e))
        end
        kw = (edgelabel=clist, kw...)
    end

    GraphPlot.gplot(g; kw...)
end
