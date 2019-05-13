function translational_symmetry_table(dims)
    nv = prod(dims)
    lat = reshape(collect(1:nv), Tuple(dims))

    iso_table = Vector{Vector{Int}}()
    if length(dims) == 1
        for i=0:dims[1]-1
            push!(iso_table, vec(circshift(lat, i)))
        end
    elseif length(dims) == 2
        for i=0:dims[1]-1
            for j=0:dims[2]-1
                push!(iso_table, vec(circshift(lat, [i,j])))
            end
        end
    elseif length(dims) == 3
        for i=0:dims[1]-1
            for j=0:dims[2]-1
                for k=0:dims[3]-1
                    push!(iso_table, vec(circshift(lat, [i,j,k])))
                end
            end
        end
    else
        @error "not supported"
    end
    iso_table
end

function all_isomorph(graph)
    isomorphs = LightGraphs.Experimental.all_isomorph(graph, graph)

    iso_table = Vector{Vector{Int}}()
    for iso=isomorphs
        map_table = zeros(Int, nv(graph))
        for (src,dst)=iso
            map_table[src] = dst
        end
        push!(iso_table, map_table)
    end
    iso_table
end
