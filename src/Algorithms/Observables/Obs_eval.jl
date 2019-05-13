function sample_network!(
  res::MCMCObsEvaluationCache,
  problem::ObservablesProblem,
  net, σ, wholespace=false)

  σp = deepcopy(σ.parent)

  # The denominator of this state
  lnψ = net(σ)
  prob = wholespace ? exp(real(lnψ)) : 1.0
  res.Zave += prob
  # Iterate through all elements in row i_σ of the matrix computing
  # Since Julia has no CSR matrices, I iterate on columns of the
  # transpose matrix
  i_σ = index(σ)

  # TODO : this works only with NDM. Should generalize it to states...
  set_index!(col(σp), i_σ)
  for (obs_id, Oᵗ) = enumerate(problem.ObservablesTransposed)
    O_loc = 0.0+0.0im
    for row_id = Oᵗ.colptr[i_σ]:(Oᵗ.colptr[i_σ+1]-1)
      # Find nonzero elements s by doing <i_sp|Oᵗ|i_σ>
      i_σp = Oᵗ.rowval[row_id]
      # BackConvert to int
      set_index!(col(σp), i_σp)
      # Compute the log(ψ(σ)/ψ(σ')), by only computing differences.
      log_ratio = net(σp) - lnψ
      O_loc += (Oᵗ.nzval[row_id]) * exp(log_ratio)
    end
    res.ObsAve[obs_id] += prob * O_loc
    push!(res.ObsVals[obs_id], prob * O_loc)
  end
  return res
end
