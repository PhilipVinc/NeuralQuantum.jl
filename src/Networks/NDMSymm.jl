export NDMSymm

struct NDMSymm{T,T2,T3} <: MatrixNeuralNetwork
    bare_net::T
    symm_net::T
    ∇b_mat::T2
    ∇h_mat::T2
    ∇d_mat::T2
    ∇w_mat::T2
    ∇u_mat::T2

    bare_der::T3
    symm_map
end

function NDMSymm(T::Type{<:Real}, n_in, αh, αa, permutations)
    n_symm = length(permutations)
    @assert length(first(permutations)) == n_in

    symm_net = NDM(T, n_in, αh//n_in,αa//n_in)
    bare_net = NDM(T, n_in, αh//n_in*n_symm, αa//n_in*n_symm)
    set_bare_params!(bare_net, symm_net, permutations)
    ∇b_mat, ∇h_mat, ∇d_mat, ∇w_mat, ∇u_mat = construct_∇matrices(bare_net, symm_net, n_in, αh, αa, permutations)
    bare_der = grad_cache(bare_net)

    NDMSymm(bare_net, symm_net, ∇b_mat, ∇h_mat, ∇d_mat, ∇w_mat, ∇u_mat, bare_der, permutations)
end

input_type(net::NDMSymm)         = input_type(net.bare_net)
weight_type(net::NDMSymm)        = out_type(net.bare_net)
out_type(net::NDMSymm)           = out_type(net.bare_net)
input_shape(net::NDMSymm)        =  input_shape(net.bare_net)
random_input_state(net::NDMSymm) = random_input_state(net.bare_net)
is_analytic(net::NDMSymm)        = is_analytic(net.bare_net)
weights(cnet::NDMSymm)           = cnet.symm_net

cache(net::NDMSymm)              = cache(net.bare_net)
grad_cache(net::NDMSymm)         = grad_cache(net.symm_net)
weight_tuple(net::NDMSymm, args...) = weight_tuple(weights(net), args...)
update!(opt, cnet::NDMSymm, Δ, state=nothing) = (res = update!(opt, weights(cnet), weights(Δ), state);
                                            set_bare_params!(cnet.bare_net, cnet.symm_net, cnet.symm_map);
                                            res)

@inline (net::NDMSymm)(args...) = net.bare_net(args...)
function logψ_and_∇logψ!(∇lnψ_symm, net::NDMSymm, args...)
    lnψ, ∇lnψ = logψ_and_∇logψ!(net.bare_der, net.bare_net, args...)

    symmetrize_∇logψ_NDM!(∇lnψ_symm, ∇lnψ, net)
    lnψ, ∇lnψ_symm
end

function symmetrize_∇logψ_NDM!(∇lnψ_symm, ∇lnψ, net)
    mul!(∇lnψ_symm.b_μ, net.∇b_mat, ∇lnψ.b_μ)
    mul!(∇lnψ_symm.b_λ, net.∇b_mat, ∇lnψ.b_λ)
    mul!(∇lnψ_symm.h_μ, net.∇h_mat, ∇lnψ.h_μ)
    mul!(∇lnψ_symm.h_λ, net.∇h_mat, ∇lnψ.h_λ)
    mul!(∇lnψ_symm.d_λ, net.∇d_mat, ∇lnψ.d_λ)
    mul!(vec(∇lnψ_symm.w_μ), net.∇w_mat, vec(∇lnψ.w_μ))
    mul!(vec(∇lnψ_symm.w_λ), net.∇w_mat, vec(∇lnψ.w_λ))
    mul!(vec(∇lnψ_symm.u_μ), net.∇u_mat, vec(∇lnψ.u_μ))
    mul!(vec(∇lnψ_symm.u_λ), net.∇u_mat, vec(∇lnψ.u_λ))
    ∇lnψ_symm
end






function set_bare_params!(bare_net, symm_net, permutations)
    # Load a bit
    n_symm = length(permutations)
    n_in   = length(symm_net.b_μ)
    αh   = size(symm_net.w_μ,1)
    αa   = size(symm_net.u_μ,1)

    # Local bias
    symm_net.b_μ .= sum(symm_net.b_μ)/length(symm_net.b_μ)
    symm_net.b_λ .= sum(symm_net.b_λ)/length(symm_net.b_λ)
    bare_net.b_μ .=  symm_net.b_μ
    bare_net.b_λ .=  symm_net.b_λ

    # Hidden biases
    for (i, (h_μ_i, h_λ_i)) = enumerate(zip(symm_net.h_μ, symm_net.h_λ))
        bare_net.h_μ[((i-1)*n_symm+1):(i*n_symm)] .=  h_μ_i
        bare_net.h_λ[((i-1)*n_symm+1):(i*n_symm)] .=  h_λ_i
    end

    # Ancilla biases
    for (i, d_λ_i) = enumerate(symm_net.d_λ)
        bare_net.d_λ[((i-1)*n_symm+1):(i*n_symm)] .=  d_λ_i
    end

    # Hidden connections
    for f=1:αh
        for (j,perm)=enumerate(permutations)
            for (i, i_p)=enumerate(perm)
                bare_net.w_μ[j + (f-1)*n_symm, i_p] = symm_net.w_μ[f,i]
                bare_net.w_λ[j + (f-1)*n_symm, i_p] = symm_net.w_λ[f,i]
            end
        end
    end

    # Ancilla connections
    for f=1:αa
        for (j,perm)=enumerate(permutations)
            for (i, i_p)=enumerate(perm)
                bare_net.u_μ[j + (f-1)*n_symm, i_p] = symm_net.u_μ[f,i]
                bare_net.u_λ[j + (f-1)*n_symm, i_p] = symm_net.u_λ[f,i]
            end
        end
    end
    bare_net
end

function construct_∇matrices(bare_net, symm_net, n_in, αh, αa, permutations)
    # Load a bit
    n_symm = length(permutations)

    # Local bias matrix
    ∇b_mat   = ones(eltype(bare_net.u_λ), n_in, n_in)./n_in

    # Hidden bias matrix
    ∇h_mat   = zeros(eltype(bare_net.h_λ), αh, αh*n_symm)
    for s=1:αh
        ∇h_mat[s, (s-1)*n_symm+1:s*n_symm] .= 1
    end

    # Ancillary bias matrix
    ∇d_mat   = zeros(eltype(bare_net.d_λ), αa, αa*n_symm)
    for s=1:αa
        ∇d_mat[s, (s-1)*n_symm+1:s*n_symm] .= 1
    end

    ∇w_mat   = zeros(eltype(bare_net.w_λ), αh*n_in, αh*n_symm*n_in)
    lin_ind_bare  = LinearIndices(bare_net.w_λ)
    lin_ind_symm  = LinearIndices(symm_net.w_λ)
    for f=1:αh
        for (j,perm)=enumerate(permutations)
            for (i, i_p)=enumerate(perm)
                row = lin_ind_symm[f,i]
                col = lin_ind_bare[j + (f-1)*n_symm, i_p]
                ∇w_mat[row,col] = 1
            end
        end
    end

    ∇u_mat   = zeros(eltype(bare_net.u_λ), αa*n_in, αa*n_symm*n_in)
    lin_ind_bare  = LinearIndices(bare_net.u_λ)
    lin_ind_symm  = LinearIndices(symm_net.u_λ)
    for f=1:αa
        for (j,perm)=enumerate(permutations)
            for (i, i_p)=enumerate(perm)
                row = lin_ind_symm[f,i]
                col = lin_ind_bare[j + (f-1)*n_symm, i_p]
                ∇u_mat[row,col] = 1
            end
        end
    end
    ∇b_mat, ∇h_mat, ∇d_mat, ∇w_mat, ∇u_mat
end

######
