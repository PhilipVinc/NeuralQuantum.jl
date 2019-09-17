# Sample network and compute observables for a set of operators stored as
# transposed sparse matrices on |ψ⟩ Neural Quantum States (KetNet).
function sample_network!(
  res::MCMCObsEvaluationCache,
  problem::ObservablesProblem,
  net::KetNet, σ, wholespace=false)
  # Load tmp caches to avoid allocating
  σp = res.σ

  # The denominator of this state
  lnψ = net(σ)
  prob = wholespace ? exp(2*real(lnψ)) : 1.0
  res.Zave += prob

  # Iterate through all elements in row i_σ of the matrix computing
  # Since Julia has no CSR matrices, I iterate on columns of the
  # transpose matrix
  i_σ = index(σ)
  for (obs_id, Oᵗ) = enumerate(problem.ObservablesTransposed)
    O_loc = 0.0+0.0im
    for row_id = Oᵗ.colptr[i_σ]:(Oᵗ.colptr[i_σ+1]-1)
      # Find nonzero elements s by doing <i_sp|Oᵗ|i_σ>
      i_σp = Oᵗ.rowval[row_id]
      set_index!(σp, i_σp)

      # Compute the log(ψ(σ)/ψ(σ')), by only computing differences.
      log_ratio = net(σp) - lnψ
      O_loc += conj(Oᵗ.nzval[row_id]) * exp(log_ratio) # conj because transpose
    end
    res.ObsAve[obs_id] += prob * O_loc
    push!(res.ObsVals[obs_id], prob * O_loc)
  end
  return res
end

# Sample network and compute observables for a set of operators stored as a
# KLocalOperator on |ψ⟩ Neural Quantum States (KetNet).
function sample_network!(
  res::MCMCObsEvaluationCache,
  problem::ObservablesProblem{B,SM},
  net::KetNet, σ, wholespace=false) where {B, SM<:AbsLinearOperator}
  # Load tmp caches to avoid allocating
  σp = res.σ

  # The denominator of this state
  lnψ = net(σ)
  prob = wholespace ? exp(2*real(lnψ)) : 1.0
  res.Zave += prob

  i_σ = index(σ)
  for (obs_id, O) = enumerate(problem.ObservablesTransposed)
    O_loc = 0.0+0.0im
    #diffs_O = row_valdiff(O, σ)
    for op=operators(O)
      r = local_index(σ, sites(op))
      for (mel, changes)=op.op_conns[r]
        set_index!(σp, i_σ)
        for (site, val)=changes
          setat!(σp, site, val)
        end

        # Compute the log(ψ(σ)/ψ(σ')), by only computing differences.
        log_ratio = net(σp) - lnψ
        O_loc += mel * exp(log_ratio)
      end
    end

    res.ObsAve[obs_id] += prob * O_loc
    push!(res.ObsVals[obs_id], prob * O_loc)
  end
  return res
end
