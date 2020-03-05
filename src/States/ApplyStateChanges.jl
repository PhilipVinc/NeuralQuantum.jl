"""
    apply!(σ, changes, [changes_r])

Applies the changes `changes` to the `σ`.

If `state isa DoubleState` then single-value changes
are applied to the columns of the state (in order to
compute matrix-operator products). Otherwise it should
be a tuple with changes of row and columns.

If the state is double but there is only 1 element of changes,
it's applied to the rows.

If changes is nothing, does nothing.
"""
apply!(σ::AbstractArray, cngs::Nothing) = σ

@inline function apply!(σ::ADoubleState, cngs_l::Union{StateChanges,Nothing}) 
    apply!(row(σ), cngs_l)
    return σ
end

@inline function apply!(σ::ADoubleState, (cngs_l, cngs_r)) 
    apply!(row(σ), cngs_l)
    apply!(col(σ), cngs_r)
    return σ
end

function apply!(σ::ADoubleState, cngs_l, cngs_r) 
    apply!(row(σ), cngs_l)
    apply!(col(σ), cngs_r)
    return σ
end

@inline apply!(σ::AbstractVector, cngs::Nothing) = σ
function apply!(σ::AbstractVector, cngs) 
    for (site, val)=cngs
        σ[site] = val
    end
    return σ
end


"""
    apply(state, cngs)

Applies the changes `cngs` to the state `σ`, by allocating a
copy.

See also @ref(apply!)
"""
apply(σ::AbstractState, cngs) = apply!(deepcopy(σ), cngs)
