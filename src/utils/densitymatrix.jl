export densitymatrix, ket

"""
    densitymatrix(net, prob, norm=true)

Returns the Density matrix encoded by the neural network `net`, and normalizes
it if `norm==true`.
"""
function densitymatrix(net::NeuralNetwork, hilb::AbstractSuperOpBasis, norm=true)
    v = state(hilb, net)

    p_hilb = physical(hilb)
    ρ = zeros(out_type(net), spacedimension(p_hilb), spacedimension(p_hilb))
    #if v isa DiagonalStateWrapper
    #    v = v.parent
    #end
    for i=1:spacedimension(p_hilb)
        set_index!(row(v), p_hilb, i)
        for j=1:spacedimension(p_hilb)
            set_index!(col(v), p_hilb, j)
            ρ[i,j] = exp(net(v))
        end
    end
    if norm
        ρ ./= tr(ρ)
    end
    ρ
end

densitymatrix(net::NeuralNetwork, hilb::AbstractHilbert, norm=true) =
    densitymatrix(net, SuperOpSpace(hilb), norm)

Base.Matrix(net::NeuralNetwork, hilb, norm=true) = densitymatrix(net, hilb, norm)
Base.Vector(net::MatrixNet, hilb, norm=true) = vec(densitymatrix(net, hilb, norm))

"""
    ket(net, prob, norm=true)

Returns the state (ket) encoded by the neural network `net`, and normalizes
it if `norm==true`.
"""
function ket(net, hilb::AbstractHilbert, norm=true)
    psi = zeros(out_type(net), spacedimension(hilb))

    v = state(hilb, net)
    for i=1:spacedimension(hilb)
        set_index!(v, hilb, i)
        psi[i] = exp(net(v))
    end
    if norm
        normalize!(psi)
    end
    return psi
end

ket(net::MatrixNet, hilb::AbstractSuperOpBasis, norm=true) =
    ket(net, hilb, norm)

ket(net::MatrixNet, hilb::AbstractHilbert, norm=true) =
    ket(net, SuperOpSpace(hilb), norm)

Base.Vector(net::NeuralNetwork, hilb, norm=true) = ket(net, hilb, norm)

QuantumOpticsBase.DenseOperator(net::NeuralNetwork, hilb::AbstractHilbert, norm=true) =
    DenseOperator(convert(CompositeBasis, hilb), Matrix(net, hilb, norm))

##
