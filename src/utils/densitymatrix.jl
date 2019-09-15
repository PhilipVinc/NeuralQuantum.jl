export densitymatrix

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

QuantumOptics.dm(net::NeuralNetwork, prob::Problem, norm=false) = densitymatrix(net, prob, norm)
