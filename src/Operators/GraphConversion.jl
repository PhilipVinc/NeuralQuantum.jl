"""
    to_linear_operator(lind::GraphLindbladian, [T=eltype(lind)])

Converts the Lindbladian to a `KLocalOperatorSum` with eltype `T` for the non
Hermitian Hamiltonian and a set of KLocalOperator(Sum) for the dissipators.

This structure has minimal memory cost, and is NEEDED when the need to represent
systems in very big lattices arises. Evidently, there is a small performance
price to pay when using those structures.
"""
to_linear_operator(lind::GraphLindbladian, T=nothing) =
    to_linear_operator(hamiltonian(lind), jump_operators_graph(lind), T)

function to_linear_operator(ham::GraphOperator, c_ops::Vector, T::Union{Nothing, Type{<:Number}}=nothing)

    ham_locs = ham.LocalOperators

    # default type
    T = isnothing(T) ?  eltype(first(ham_locs).data) : T

    op_loc = KLocalOperatorRow(T, [1], [length(basis(first(ham_locs)))],
                               first(ham_locs).data)

    H = KLocalOperatorSum(op_loc)

    # Add the local hamiltonian terms
    for (i, h_loc) = enumerate(ham.LocalOperators)
        i == 1 && continue

        dim = length(basis(h_loc))
        sum!(H, KLocalOperatorRow(T, [i], [dim], h_loc.data))
    end

    # Hoppings
    for edge=edges(graph(ham))
        connections = ham.EdgeOpList[edge]
        for (coeff, l_id, r_id)=connections
            hl = ham.HopOperatorsList[l_id]
            hr = ham.HopOperatorsList[r_id]

            dim_l = length(basis(hl))
            dim_r = length(basis(hr))
            sum!(H, KLocalOperatorRow(T, [edge.src, edge.dst],
                                      [dim_l, dim_r],
                                      coeff*((hl⊗hr).data)))
        end
    end

    # store the vector of local-acting, individual dissipators
    loss_ops   = []
    loss_ops_t = []

    # dissipations
    # TODO support nonlocal dissipators
    for (i, L) = enumerate(c_ops)

        # Find local basis where the operator is non-zero
        nz_sites = Int[]
        for (j, L_j) = enumerate(L.LocalOperators)
            length(L_j.data.nzval) == 0 && continue
            push!(nz_sites, j)
        end

        isempty(nz_sites) && continue

        hilb_dims = length.(basis.(L.LocalOperators[nz_sites]))

        L_nz      = tensor(L.LocalOperators[nz_sites]...)

        op        = KLocalOperatorRow(T, nz_sites, hilb_dims, L_nz.data)

        op_hnh    = KLocalOperatorRow(T, nz_sites, hilb_dims,
                                      -im/2*(L_nz'*L_nz).data)

        sum!(H, op_hnh)
        push!(loss_ops, op)
        push!(loss_ops_t, transpose(op))
    end

    T = isempty(loss_ops) ? Nothing : typeof(first(loss_ops))

    loss_ops   = Vector{T}(loss_ops)
    loss_ops_t = Vector{T}(loss_ops_t)

    return (H, loss_ops, loss_ops_t)
end

"""
    to_linear_operator(lind::GraphLindbladian, [T=eltype(lind)])

Converts the Hamiltonian to a `KLocalOperatorSum` with eltype `T`.

This structure has minimal memory cost, and is NEEDED when the need to represent
systems in very big lattices arises. Evidently, there is a small performance
price to pay when using those structures.
"""
function to_linear_operator(op::GraphOperator, T::Union{Nothing, Type{<:Number}}=nothing)

    op_locs = op.LocalOperators

    # default type
    T = isnothing(T) ?  eltype(first(op_locs).data) : T

    op_loc = KLocalOperatorRow(T, [1], [length(basis(first(op_locs)))],
                        first(op_locs).data)

    res_op = KLocalOperatorSum(op_loc)

    # Add the local hamiltonian terms
    for (i, op_loc) = enumerate(op.LocalOperators)
        i == 1 && continue # the first one was already added by the constructor.

        dim = length(basis(op_loc))
        sum!(res_op, KLocalOperatorRow(T, [i], [dim], op_loc.data))
    end

    # Hoppings
    for edge=edges(graph(op))
        edge ∉ keys(op.EdgeOpList) && continue

        connections = op.EdgeOpList[edge]
        for (coeff, l_id, r_id)=connections
            hl = op.HopOperatorsList[l_id]
            hr = op.HopOperatorsList[r_id]

            dim_l = length(basis(hl))
            dim_r = length(basis(hr))
            sum!(res_op, KLocalOperatorRow(T, [edge.src, edge.dst],
                                           [dim_l, dim_r],
                                           coeff*((hl⊗hr).data)))
        end
    end

    return res_op
end
