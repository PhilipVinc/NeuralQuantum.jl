to_linear_operator(lind::GraphLindbladian) =
    to_linear_operator(hamiltonian(lind), jump_operators_graph(lind))

function to_linear_operator(ham::GraphOperator, c_ops::Vector)

    ham_locs = ham.LocalOperators
    op_loc = KLocalOperatorRow([1], [length(basis(first(ham_locs)))],
                        first(ham_locs).data)

    H = KLocalOperatorSum(op_loc)

    # Add the local hamiltonian terms
    for (i, h_loc) = enumerate(ham.LocalOperators)
        i == 1 && continue

        dim = length(basis(h_loc))
        sum!(H, KLocalOperatorRow([i], [dim], h_loc.data))
    end

    # Hoppings
    for edge=edges(graph(ham))
        connections = ham.EdgeOpList[edge]
        for (coeff, l_id, r_id)=connections
            hl = ham.HopOperatorsList[l_id]
            hr = ham.HopOperatorsList[r_id]

            dim_l = length(basis(hl))
            dim_r = length(basis(hr))
            sum!(H, KLocalOperatorRow([edge.src, edge.dst],
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

        hilb_dims = length.(basis.(L.LocalOperators[nz_sites]))

        L_nz      = tensor(L.LocalOperators[nz_sites]...)

        op        = KLocalOperatorRow(nz_sites, hilb_dims, L_nz.data)

        op_hnh    = KLocalOperatorRow(nz_sites, hilb_dims,
                                      -im*(L_nz'*L_nz).data)

        sum!(H, op_hnh)
        push!(loss_ops, op)
        push!(loss_ops_t, transpose(op))
    end

    loss_ops   = Vector{typeof(first(loss_ops))}(loss_ops)
    loss_ops_t = Vector{typeof(first(loss_ops))}(loss_ops_t)

    return (H, loss_ops, loss_ops_t)
end
