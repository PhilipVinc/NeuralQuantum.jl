## Matrix whole space
function sample_network!(res::MCMCGradientEvaluationCache,
  prob::HermitianMatrixProblem, net, σ, wholespace=false)

  lnψ, ∇lnψ = logψ_and_∇logψ!(res.∇lnψ, net, σ)
  E         = compute_Cloc(prob, net, σ, lnψ, res.σ)

  prob = wholespace ? exp(2*real(lnψ)) : 1.0
  res.Zave   += prob
  res.Eave   += prob  * E
  push!(res.Evalues, prob*E)

  for (i, _∇lnψ)= enumerate(∇lnψ.tuple_all_weights)
    res.Oave[i]  .+= prob .* _∇lnψ
    res.EOave[i] .+= prob .* E .* conj.(_∇lnψ)
  end
  return res
end

## Matrix whole space
function sample_network!(res::MCMCGradientLEvaluationCache, prob::LRhoSquaredProblem,
                         net, σ, wholespace=false)
  CLO_i = res.LLO_i

  lnψ, ∇lnψ = logψ_and_∇logψ!(res.∇lnψ, net, σ)
  C_loc = compute_Cloc!(CLO_i, res.∇lnψ2, prob, net, σ, lnψ, res.σ)

  prob = wholespace ? exp(2*real(lnψ)) : 1.0
  E = abs(C_loc)^2

  res.Zave += prob
  res.Eave += prob * E
  push!(res.Evalues, prob*E)

  for (i, _∇lnψ)= enumerate(∇lnψ.tuple_all_weights)
    res.Oave[i]  .+= prob .* _∇lnψ
    res.EOave[i] .+= prob .* E .* conj.(_∇lnψ)
    res.LLOave[i].+= prob .* conj(C_loc) .* CLO_i[i]
  end
  return res
end
