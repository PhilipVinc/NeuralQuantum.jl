export densitymatrix, ket

"""
    densitymatrix(net, prob, norm=true)

Returns the Density matrix encoded by the neural network `net`, and normalizes
it if `norm==true`.
"""
function densitymatrix(net, prob, norm=true)
    ρ = DenseOperator(basis(prob))
    v = state(prob, net)
    if v isa DiagonalStateWrapper
        v = v.parent
    end
    for i=1:spacedimension(row(v))
        set_index!(row(v), i)
        for j=1:spacedimension(col(v))
            set_index!(col(v),j)
            ρ.data[i,j] = exp(net(v))
        end
    end
    if norm
        ρ.data ./= tr(ρ)
    end
    ρ
end

QuantumOptics.dm(net::NeuralNetwork, prob::AbstractProblem, norm=false) =
    densitymatrix(net, prob, norm)

"""
    ket(net, prob, norm=true)

Returns the state (ket) encoded by the neural network `net`, and normalizes
it if `norm==true`.
"""
function ket(net, prob, norm=true)
    psi = Ket(basis(prob))
    v = state(prob, net)
    for i=1:spacedimension(v)
        set_index!(v, i)
        psi.data[i] = exp(net(v))
    end
    if norm
        normalize!(psi)
    end
    return psi
end
