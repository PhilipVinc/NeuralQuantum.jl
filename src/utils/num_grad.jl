export gradient_numerical

_expval(psi::AbstractVector, Ĥ, op::AbsLinearOperator) =
    psi'*Ĥ*psi/(psi'psi)

_expval(psi::AbstractVector, Ĥ, op::KLocalLiouvillian) =
    sum(abs2.((Ĥ*psi)./psi))


function gradient_numerical(net, op, h=0.001)
    Ĥ = Matrix(op)
    cnet = cached(net)
    hilb = basis(op)
    psi = Vector(cnet, hilb)

    ∇g = grad_cache(net)
    ∇gv = vec_data(∇g)[1]
    Δ = grad_cache(eltype(trainable_first(net)), net)
    Δv = vec_data(Δ)[1]

    optimizer = Optimisers.Descent(1)

    if eltype(Δv) isa Complex
        h = h+h*1im
    end

    Δv .= 0.0
    for i=1:length(Δv)
        Δv[i] = -h
        Optimisers.update!(optimizer, net, Δ)
        E_p = _expval(Vector(cnet, hilb), Ĥ, op)

        Δv[i] = 2h
        Optimisers.update!(optimizer, net, Δ)
        E_m = _expval(Vector(cnet, hilb), Ĥ, op)

        Δv[i] = -h
        Optimisers.update!(optimizer, net, Δ)
        Δv[i] = 0

        ∇gv[i] = (E_p - E_m) / (2h)
    end
    return ∇g
end
