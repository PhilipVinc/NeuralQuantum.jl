using LightGraphs.SimpleGraphs: SimpleEdge, AbstractSimpleGraph

mutable struct HyperCube{T,SG<:AbstractGraph{T}} <: AbstractWrappedSimpleGraph{T}
    bare_graph::SG

    dims::Vector{T}
    vert_coords::Vector{Vector{T}}
end

"""
    HyperCube([dims::Int], periodic=true) -> HyperCube

Creates a HyperCube graph with spatial dimensions specified by dims. If `periodic==true`
then every dimension is periodic. You can also pass `periodic=[true, false, false]`
and only the specified dimensions will be periodic

The returned graph holds the lattice coordinates of every vertex in the underlying
graph, useful when computing correlation functions.
"""
function HyperCube(dims, periodic=true, neighbours_distances=[1])
    T=Int

    nv = prod(dims)
    vert_coords = Vector{Vector{T}}()

    # Convert the linear indices of vertices to N-dim coordinates
    for i=1:nv
        push!(vert_coords, int_to_hypercube_coord(i, dims))
    end

    #
    if !(periodic isa AbstractVector)
        periodic = fill(periodic, size(dims))
    elseif size(periodic) ≠ size(dims)
        throw("Error: periodic should match sizes")
    end

    sg = SimpleGraph{T}(nv)

    # If neighbours distances is an integer use only this as a distance
    if neighbours_distances isa Integer
        neighbours_distances = [neighbours_distances]
    end

    for dist = neighbours_distances
        is_good = all(dist .< dims)
        if !is_good
            throw("neighbours_distances should be greater than all dimensions.")
        end
    end

    if length(neighbours_distances) > 1
        sg = ColoredGraph(sg)
    end

    _add_edge!(sg::AbstractGraph, src, dst, col) = add_edge!(sg, src, dst)
    _add_edge!(sg::ColoredGraph, src, dst, col) = add_edge!(sg, src, dst, col)

    for i=1:nv
        neighbors = Vector{T}()
        coord_o   = vert_coords[i]
        for (j, (dim, loop_dim)) = enumerate(zip(dims, periodic))
            for (col,Δ)=enumerate(neighbours_distances)
                for δⱼ=[Δ,-Δ]
                    nj = coord_o[j]+δⱼ
                    if (nj < 1 || nj > dim)
                        # if we don't loop this dimension lets skip this connection
                        !loop_dim && continue

                        # wrap around
                        nj = nj<=0 ? rem(nj,dim) + dim : rem(nj,dim)
                    end

                    coord = copy(coord_o)
                    coord[j] = nj
                    push!(neighbors, findfirst(x->x==coord, vert_coords))
                    _add_edge!(sg, i, findfirst(x->x==coord, vert_coords), col)
                end
            end
        end
    end

    return HyperCube(sg, dims, vert_coords)
end

## make it static


# I/O
show(io::IO, g::HyperCube) = print(io, "HyperCube with $(nv(g)) vertices, $(ne(g)) edges")
######

int_to_hypercube_coord(r, dims::Vector) = int_to_hypercube_coord(r, Tuple(dims)...)
int_to_hypercube_coord(c, dims...) = begin
    r, cc = divrem(c-1,prod(Base.front(dims)))
    append!(int_to_hypercube_coord(cc+1, Base.front(dims)...), r+1)
end
int_to_hypercube_coord(c, dims::Vararg{Integer, 1}) = [c]


## Useful functions
translational_symm_table(g::HyperCube) = _hypercube_translational_symmetry_table(g.dims)
function _hypercube_translational_symmetry_table(dims)
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
    return iso_table
end
