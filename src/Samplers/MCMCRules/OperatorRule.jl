export OperatorRule

"""
    OperatorRule(Ô)

Transition rule for Metropolis-Hastings sampling where at every step a move is
drawn from those allowed by the Operator `Ô`.

Couples of sites are generated from the graph `graph` (or operator), where all
sites that are connected (or coupled by a 2-body term) are considered for switches.

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

    i = rand(1:n_forward)
    apply!(σp, conns[i])

    row_valdiff!(conns, Ô, σp, init=true)
    n_back = length(conns)

    c.log_prob_bias = log(n_forward/n_back)
end

function propose_step!(σp::Union{AStateBatch,ADoubleStateBatch}, s::MetropolisSampler{<:OperatorRule},
                       net::Union{MatrixNet,KetNet}, c, rc)
    Ô     = s.rule.operator
    conns = rc.conns

    for b=1:num_batches(σp)
        σp_b = unsafe_get_batch(σp, b)

        row_valdiff!(conns, Ô, σp_b, init=true)
        n_forward = length(conns)

        i = rand(1:n_forward)
        mel, cngs = conns[i]
        apply!(σp_b, cngs)

        row_valdiff!(conns, Ô, σp_b, init=true)
        n_back = length(conns)

        c.log_prob_bias[b] = log(n_forward/n_back)
    end
end
