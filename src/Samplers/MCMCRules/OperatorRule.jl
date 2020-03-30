export OperatorRule

"""
    OperatorRule(Ô)

Transition rule for Metropolis-Hastings sampling where at every step a move is
drawn from those allowed by the Operator `Ô`.
"""
struct OperatorRule{O} <: MCMCRule
    operator::O
end

struct OperatorRuleCache{O}
    conns::O
end

function RuleSamplerCache(rule::OperatorRule, s, v, net, par_cache)
    state = rand(basis(rule.operator))
    conns = row_valdiff(rule.operator, state)
    return OperatorRuleCache(conns)
end

function propose_step!(σp::Union{AState,ADoubleState}, s::MetropolisSampler{<:OperatorRule},
                       net::Union{MatrixNet,KetNet}, c, rc)
    Ô     = s.rule.operator
    conns = rc.conns

    row_valdiff!(conns, Ô, σp, init=true)
    n_forward = length(conns)

    mel, cngs = rand(conns)
    @inbounds apply!(σp, cngs)

    row_valdiff!(conns, Ô, σp, init=true)
    n_back = length(conns)

    c.log_prob_bias = log(n_forward/n_back)
end


# Don't implement the batched version and use default fallback
function propose_step!(σp::Union{AStateBatch,ADoubleStateBatch}, s::MetropolisSampler{<:OperatorRule},
                       net::Union{MatrixNet,KetNet}, c, rc)
    Ô     = s.rule.operator
    conns = rc.conns

    for (i,σpᵢ)=enumerate(states(σp))
        row_valdiff!(conns, Ô, σpᵢ, init=true)
        n_forward = length(conns)

        mel, cngs = rand(conns)
        @inbounds apply!(σpᵢ, cngs)

        row_valdiff!(conns, Ô, σpᵢ, init=true)
        n_back = length(conns)

        c.log_prob_bias[i] = log(n_forward/n_back)
    end
end
