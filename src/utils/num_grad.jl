export gradient_numerical

function gradient_numerical(net::KetNet, op, h=0.001)
    Ĥ = Matrix(op)
    cnet = cached(net)
    hilb = basis(op)
    psi = Vector(cnet, hilb)

    ∇g = grad_cache(net)
    ∇gv = vec_data(∇g)[1]
    Δ = grad_cache(eltype(trainable_first(net)), net)
    Δv = vec_data(Δ)[1]

    optimizer = Optimisers.Descent(h)

    expval(psi::AbstractVector) =
        psi'*Ĥ*psi/(psi'psi)

    if eltype(Δv) isa Complex
        h = h+h*1im
    end

    Δv .= 0.0
    for i=1:length(Δv)
        Δv[i] = -h
        Optimisers.update!(optimizer, net, Δ)
        E_p = expval(Vector(cnet, hilb))

        Δv[i] = 2h
        Optimisers.update!(optimizer, net, Δ)
        E_m = expval(Vector(cnet, hilb))

        Δv[i] = -h
        Optimisers.update!(optimizer, net, Δ)

        ∇gv[i] = (E_p - E_m) / (2h)
    end
    return ∇g
end
