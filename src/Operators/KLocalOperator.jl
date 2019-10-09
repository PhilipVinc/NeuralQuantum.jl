"""
    KLocalOperator

A representation of a local operator, acting on certain `sites`, each with
`hilb_dims` dimension. The local operator is written in matrix form in this
basis as `mat`. For every row of `mat` there are several non-zero values contained
in `mel`, and to each of those, `to_change` contains the sites that must change the basis
new value, contained in `new_value`
"""
struct KLocalOperator{SV,M,Vel,Vti,Vtc,Vtv,OC} <: AbsLinearOperator
    # list of sites on which this operator acts
    sites::SV

    # local hilbert space dimensions for each site
    hilb_dims::SV

    # Operator (as a matrix) represented
    mat::M

    # Matrix elements, per row
    mel::Vector{Vel}

    # List of indices that should be changed
    to_change::Vector{Vtc}

    # List of values of that should be changed
    new_values::Vector{Vtv}

    # Connections
    op_conns::Vector{OC}

    # List of state indices corresponding to mel
    # This is redundant, and equivalent to specifying to_change and new_values
    new_indices::Vector{Vti}
end

"""
    KLocalOperatorRow(sites::Vector, hilb_dims::Vector, operator)

Creates a KLocalOperator where connections are stored by row for the operator
    `operator` acting on `sites` each with `hilb_dims` local dimension.
"""
KLocalOperatorRow(sites::AbstractVector, hilb_dims::AbstractVector, operator) =
    KLocalOperatorRow(eltype(operator), sites, hilb_dims, operator)

function KLocalOperatorRow(T::Type{<:Number}, sites::AbstractVector, hilb_dims::AbstractVector, operator)
    # TODO Generalize to arbistrary spaces and not only uniform
    st  = NAryState(real(T), first(hilb_dims), length(sites))
    st1 = NAryState(real(T), first(hilb_dims), length(sites))

    mel         = Vector{Vector{T}}()
    new_indices = Vector{Vector{Int}}()
    to_change   = Vector{Vector{Vector{eltype(Int)}}}()
    new_values  = Vector{Vector{Vector{eltype(st)}}}()
    op_conns    = Vector{OpConnection{Vector{T},
                        Vector{eltype(Int)}, Vector{eltype(st)}}}()
    #SCT = StateChanges{eltype(to_change_els), eltype(new_values_els)}

    for (r, row) = enumerate(eachrow(operator))
        # create the containers for this row
        mel_els        = T[]
        mel_indices    = Int[]
        to_change_els  = Vector{Vector{Int}}()
        new_values_els = Vector{Vector{T}}()
        conns_els      = eltype(op_conns)()

        if abs(row[r]) > 10e-6
            push!(mel_els, row[r])
            push!(mel_indices, r)
            push!(to_change_els, Int[])
            push!(new_values_els, eltype(st)[])
            push!(conns_els, (row[r], Int[], eltype(st)[]))
        end

        set_index!(st, r)
        for (c, val) = enumerate(row)
            r == c && continue
            abs(val) < 10e-6 && continue

            set_index!(st1, c)
            cngd = Int[]
            nwvls = eltype(st)[]
            for (i, (v, vn)) = enumerate(zip(config(st), config(st1)))
                if v != vn
                    push!(cngd, sites[i])
                    push!(nwvls, vn)
                end
            end
            push!(mel_els, val)
            push!(mel_indices, c)
            push!(to_change_els, cngd)
            push!(new_values_els, nwvls)
            push!(conns_els, (val, cngd, nwvls))
        end

        push!(mel, mel_els)
        push!(new_indices, mel_indices)
        push!(to_change, to_change_els)
        push!(new_values, new_values_els)
        push!(op_conns, conns_els)
    end
    KLocalOperator(sites, hilb_dims, convert(Matrix{T}, operator |> collect),
                        mel, to_change, new_values, op_conns, new_indices)
end

KLocalOperator(op::KLocalOperator, mat::AbstractMatrix) =
    KLocalOperatorRow(copy(sites(op)),  copy(hilb_dims(op)), mat)


## Accessors
"""
    sites(op::KLocalOperator)

Returns the vector of `Int` labelling the lattice sites on which this operator
acts, in no particular order.
"""
sites(op::KLocalOperator) = op.sites

hilb_dims(op::KLocalOperator) = op.hilb_dims

operators(op::KLocalOperator) = (op,)

conn_type(top::Type{KLocalOperator{SV,M,Vel,Vti,Vtc,Vtv,OC}}) where {SV, M, Vel, Vti, Vtc, Vtv, OC} =
    OpConnection{Vel, eltype(Vtc), eltype(Vtv)}
conn_type(op::KLocalOperator{SV,M,Vel,Vti,Vtc,Vtv,OC}) where {SV, M, Vel, Vti, Vtc, Vtv, OC} =
    OpConnection{Vel, eltype(Vtc), eltype(Vtv)}

# Copy
function duplicate(op::KLocalOperator)
    KLocalOperator(deepcopy(op.sites), deepcopy(op.hilb_dims), deepcopy(op.mat),
                    deepcopy(op.mel), deepcopy(op.to_change),
                    deepcopy(op.new_values), deepcopy(op.op_conns),
                    deepcopy(op.new_indices))
end

##
function row_valdiff!(conn::OpConnection, op::KLocalOperator, v::State)
    # Find row index
    r = local_index(v, sites(op))

    append!(conn, op.op_conns[r])
end

function row_valdiff_index!(conn::OpConnectionIndex, op::KLocalOperator, v::State)
    # Find row index
    r = local_index(v, sites(op))

    # Return values
    mel = op.mel[r]
    ids = op.new_indices[r]

    append!(conn, (mel, ids))
end

function accumulate_connections!(acc::AbstractAccumulator, op::KLocalOperator, v::State)
    # If it is a doublestate, we are probably computing Operator x densitymatrix,
    # so we only iterate along the column of v
    if v isa DoubleState
        r = local_index(col(v), sites(op))
    else
        r = local_index(v, sites(op))
    end

    for (mel,changes)=op.op_conns[r]
        acc(mel, changes, v)
    end

    return acc
end

_sum_samesite(op_l::KLocalOperator, op_r::KLocalOperator) = _sum_samesite!(duplicate(op_l), op_r)

function _sum_samesite!(op_l::KLocalOperator, op_r::KLocalOperator)
    @assert op_l.sites == op_r.sites
    @assert length(op_l.mel) == length(op_r.mel)

    op_l.mat .+= op_r.mat

    n = length(op_l.mel)
    for (i, (mel_vals_l, mel_vals_r)) = enumerate(zip(op_l.mel, op_r.mel))

        for (mel, ids, tch, nv) = zip(op_r.mel[i], op_r.new_indices[i],
                                      op_r.to_change[i], op_r.new_values[i])
            found = 0
            # if mel_val_l is empty, then for sure there is nothing in here
            if !isempty(mel_vals_l)
                ids_tch = findall(isequal(tch), op_l.to_change[i])
                for id=ids_tch
                    if op_l.new_values[i][id] == nv
                        found = id
                        break
                    end
                end
            end

            if found == 0
                push!(op_l.mel[i], mel)
                push!(op_l.new_indices[i], ids)
                push!(op_l.to_change[i], tch)
                push!(op_l.new_values[i], tch)
            else
                op_l.mel[i][found] += mel
            end
        end
    end

    for (i, (connsᵢ_l , connsᵢ_r)) = enumerate(zip(op_l.op_conns, op_r.op_conns))
        add!(connsᵢ_l, connsᵢ_r)
    end
    return op_l
end

Base.transpose(op::KLocalOperator) =
    KLocalOperator(op, collect(transpose(op.mat)))

function Base.conj!(op::KLocalOperator)
    conj!(op.mat)
    map(conj!, op.op_conns)
    for el=op.mel
        conj!(el)
    end
    return op
end

Base.conj(op::KLocalOperator) = conj!(duplicate(op))
Base.adjoint(op::KLocalOperator) = conj(transpose(op))

*(a::Number, b::KLocalOperator) =
    _op_alpha_prod(b,a)
*(b::KLocalOperator, a::Number) =
    _op_alpha_prod(b,a)
_op_alpha_prod(op::KLocalOperator, a::Number) =
    KLocalOperator(op, a*op.mat)

function *(opl::KLocalOperator, opr::KLocalOperator)
    if sites(opl) == sites(opr)
        return KLocalOperator(opl, opl.mat * opr.mat)
    else
        disjoint = true
        for s=sites(opr)
            if s ∈ sites(opl)
                disjoint = false
                break
            end
        end
        if disjoint
            _kop_kop_disjoint_prod(opl,opr)
        else
            _kop_kop_joint_prod(opl, opr)
        end
    end
end

function _kop_kop_disjoint_prod(opl::KLocalOperator, opr::KLocalOperator)
    sl = sites(opl); sr = sites(opr)
    if length(sl) == 1 && length(sr) == 1
        # it's commutative
        if sl[1] > sr[1]
            _op =opl
            opl = opr
            opr = _op
        end
        sl = first(sites(opl))
        sr = first(sites(opr))

        hilb_dim_l = first(hilb_dims(opl))
        hilb_dim_r = first(hilb_dims(opr))

        # inverted also in QuantumOptics... who knows why
        mat = kron(opr.mat, opl.mat)
        return KLocalOperatorRow([sl, sr], [hilb_dim_l, hilb_dim_r], mat)
    else
        sites_new = sort(vcat(sites(opl), sites(opr)))
        ids_l = [findfirst(i .==sites_new) for i=sites(opl)]
        ids_r = [findfirst(i .==sites_new) for i=sites(opr)]
        throw("to implement error $(sites(opl)) and $(sites(opr))")
    end
end

function _kop_kop_joint_prod(opl::KLocalOperator, opr::KLocalOperator)
    sl = sites(opl); sr = sites(opr)
    if length(sl) == 1 || length(sr) == 1
        reversed = false
        if length(sl) == 1
            _op = opl
            opl = opr
            opr = _op
            reversed = true
        end
        # opl has many dims, opr only 1
        sr = first(sites(opr))
        r_index = findfirst(sr .== sl)
        hdim_r = first(hilb_dims(opr))

        matrices = [Matrix(I, d, d) for d=hilb_dims(opl)]
        matrices[r_index] = opr.mat
        mat_r = kron(matrices...)
        prod_mat = reversed ? mat_r*opl.mat : opl.mat*mat_r

        return KLocalOperator(opl, prod_mat)
    else
        sites_new = sort(vcat(sites(opl), sites(opr)))
        ids_l = [findfirst(i .==sites_new) for i=sites(opl)]
        ids_r = [findfirst(i .==sites_new) for i=sites(opr)]
        throw("to implement error")
    end
end

function permutesystems(a::AbstractMatrix, h_dims::Vector, perm::Vector{Int})
    #@assert length(a.basis_l.bases) == length(a.basis_r.bases) == length(perm)
    #@assert isperm(perm)
    data = reshape(a, [h_dims; h_dims]...)
    data = permutedims(data, [perm; perm .+ length(perm)])
    data = reshape(data, prod(h_dims), prod(h_dims))
    return data
end

Base.eltype(::T) where {T<:KLocalOperator} = eltype(T)
Base.eltype(T::Type{KLocalOperator{SV,M,Vel,Vti,Vtc,Vtv,OC}}) where {SV,M,Vel,Vti,Vtc,Vtv,OC} =
    eltype(eltype(Vel))
