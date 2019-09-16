# Sample network and compute observables for observables stored as
# sparse matrices, therefore where we have a list of non-zero elements in the
# column.
function sample_network!(
  res::MCMCObsEvaluationCache,
  problem::ObservablesProblem,
  net::MatrixNet, σ, wholespace=false)

  σp = parent(res.σ)

  # The denominator of this state
  lnψ = net(σ)
  prob = wholespace ? exp(real(lnψ)) : 1.0
  res.Zave += prob
  # Iterate through all elements in row i_σ of the matrix computing
  # Since Julia has no CSR matrices, I iterate on columns of the
  # transpose matrix
  i_σ = index(σ)

  set_index!(row(σp), i_σ)
  set_index!(col(σp), i_σ)

  # TODO : this works only with NDM. Should generalize it to states...
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

# Sample network and compute observables for a KLocalOperator representation
# of observables without using lookuptables
#
function sample_network!(
  res::MCMCObsEvaluationCache,
  problem::ObservablesProblem{B,SM},
  net::MatrixNet, σ, wholespace=false) where {B, SM<:AbsLinearOperator}

  σp = parent(res.σ)

  # The denominator of this state
  lnψ = net(σ)
  prob = wholespace ? exp(real(lnψ)) : 1.0
  res.Zave += prob
  # Iterate through all elements in row i_σ of the matrix computing
  # Since Julia has no CSR matrices, I iterate on columns of the
  # transpose matrix
  i_σ = index(σ)

  set_index!(row(σp), i_σ)
  set_index!(col(σp), i_σ)

  # TODO : this works only with NDM. Should generalize it to states...
  for (obs_id, O) = enumerate(problem.ObservablesTransposed)
    O_loc = 0.0+0.0im
    #diffs_O = row_valdiff(O, row(σ.parent))
    for op=operators(O)
      r=local_index(row(σ.parent), sites(op))
      for (mel, changes)=op.op_conns[r]
        set_index!(col(σp), i_σ)
        for (site, val)=changes
          setat!(σp, site, val)
        end
        # Compute the log(ψ(σ)/ψ(σ')), by only computing differences.
        log_ratio = net(σp) - lnψ
        O_loc += conj(mel) * conj(exp(log_ratio)) # maybe a conj
      end
    end

    res.ObsAve[obs_id] += prob * O_loc
    push!(res.ObsVals[obs_id], prob * O_loc)
  end
  return res
end

function sample_network!(
  res::MCMCObsEvaluationCache,
  problem::ObservablesProblem{B,SM},
  net::MatrixNet, σ::LUState, wholespace=false) where {B, SM<:AbsLinearOperator}

  # The denominator of this state
  prob = wholespace ? exp(real(net(σ))) : 1.0
  res.Zave += prob
  #i_σ = index(σ)
  σs = state(σ)
  no_changes = changes(row(𝝝s))

  for (obs_id, O) = enumerate(problem.ObservablesTransposed)
    O_loc = 0.0+0.0im
    diffs_O = row_valdiff(O, row(σs))
    for (mel, changes)=diffs_O
      # Compute the log(ψ(σ)/ψ(σ')), by only computing differences.
      log_ratio = Δ_logψ(net, σ, changes, no_changes)
      O_loc += mel * exp(log_ratio)
    end

    res.ObsAve[obs_id] += prob * O_loc
    push!(res.ObsVals[obs_id], prob * O_loc)
  end
  return res
end
