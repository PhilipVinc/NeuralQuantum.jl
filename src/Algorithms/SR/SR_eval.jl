
## Matrix sampled
function sample_network!(
  res::MCMCSREvaluationCache,
  prob::HermitianMatrixProblem,
  net, σ, wholespace=false)

  T = real(out_type(net))
  lnψ, ∇lnψ = logψ_and_∇logψ!(res.∇lnψ, net, σ)
  E         = compute_Cloc(prob, net, σ, lnψ, res.σ)

  prob        = wholespace ? exp(T(2)*real(lnψ)) : one(T)
  res.Eave   += prob * E
  res.Zave   += prob #1.0 #exp(2*real(lnψ))
  push!(res.Evalues, prob*E)

  for (i,_∇lnψ)=enumerate(∇lnψ.tuple_all_weights)
    res.Oave[i]  .+= prob .* _∇lnψ
    res.OOave[i] .+= prob .* conj.(_∇lnψ) .* transpose(_∇lnψ)
    res.EOave[i] .+= prob .* conj.(_∇lnψ) .* E
  end
  return res
end

function sample_network!(res::MCMCSRLEvaluationCache,
                         prob::LRhoSquaredProblem,
                         net, σ, wholespace=false)
  T = real(out_type(net))
  CLO_i = res.LLO_i

  lnψ, ∇lnψ = logψ_and_∇logψ!(res.∇lnψ, net, σ)
  C_loc = compute_Cloc!(CLO_i, res.∇lnψ2, prob, net, σ, lnψ, res.σ)

  prob = wholespace ? exp(T(2)*real(lnψ)) : one(T)
  E = abs(C_loc)^2

  res.Zave += prob
  res.Eave += prob * E
  push!(res.Evalues, prob*E)

  for (i, _∇lnψ)= enumerate(∇lnψ.tuple_all_weights)
    res.Oave[i]  .+= prob .* _∇lnψ
    res.OOave[i] .+= prob .* conj.(_∇lnψ) .* transpose(_∇lnψ)
    res.EOave[i] .+= prob .* conj.(_∇lnψ) .* E
    res.LLOave[i].+= prob .* conj.(C_loc)  .* CLO_i[i]
  end
  return res
end

#=
function sample_network!(res::MCMCSREvaluationCache,
                         prob::OpenTimeEvolutionProblem,
                         net, σ, wholespace=false)
  lnψ, ∇lnψ = logψ_and_∇logψ!(res.∇lnψ, net, σ)
  E         = compute_Cloc(prob, net, σ, lnψ)

  prob = exp(2*real(lnψ))
  res.Zave   += prob
  res.Eave   += prob  * E
  push!(res.Evalues, prob*E)

  for (i, _∇lnψ)= enumerate(∇lnψ.tuple_all_weights)
    res.Oave[i]  .+= prob .* _∇lnψ
    res.OOave[i] .+= prob .* conj.(_∇lnψ) .* transpose(_∇lnψ)
    res.EOave[i] .+= prob .* conj.(_∇lnψ) .* E
  end
  return res
end=#
