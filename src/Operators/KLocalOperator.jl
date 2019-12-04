"""
    KLocalOperator

A representation of a local operator, acting on certain `sites`, each with
`hilb_dims` dimension. The local operator is written in matrix form in this
basis as `mat`. For every row of `mat` there are several non-zero values contained
in `mel`, and to each of those, `to_change` contains the sites that must change the basis
new value, contained in `new_value`
"""
struct KLocalOperator{H<:AbstractHilbert,SV,M,Vel,Vti,Vtc,Vtv,OC} <: AbsLinearOperator
    hilb::H

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
#KLocalOperatorRow(sites::AbstractVector, hilb_dims::AbstractVector, operator) = begin
#    hilb = DiscreteHilbert(hilb_dims)
#    KLocalOperatorRow(hilb, eltype(operator), sites, hilb_dims, operator)
#end

KLocalOperatorRow(hilb::AbstractHilbert, sites::AbstractVector, operator) =
    KLocalOperatorRow(eltype(operator), hilb, sites, shape(hilb)[sites], operator)


function KLocalOperatorRow(T::Type{<:Number}, hilb::AbstractHilbert, sites::AbstractVector, hilb_dims::AbstractVector, operator)
    # TODO Generalize to arbistrary spaces and not only uniform
    st  = state(T, hilb)
    st1 = state(T, hilb)

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

        set!(st, hilb, r)
        for (c, val) = enumerate(row)
            r == c && continue
            abs(val) < 10e-6 && continue

            set!(st1, hilb, c)
            cngd = Int[]
            nwvls = eltype(st)[]
            for (i, (v, vn)) = enumerate(zip(st, st1))
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
    KLocalOperator(hilb, sites, hilb_dims, convert(Matrix{T}, operator |> collect),
                        mel, to_change, new_values, op_conns, new_indices)
end

KLocalOperator(op::KLocalOperator, mat::AbstractMatrix) =
    KLocalOperatorRow(basis(op), copy(sites(op)), mat)

## Accessors
"""
    sites(op::KLocalOperator)

Returns the vector of `Int` labelling the lattice sites on which this operator
acts, in no particular order.
"""
sites(op::KLocalOperator) = op.sites

hilb_dims(op::KLocalOperator) = op.hilb_dims

QuantumOpticsBase.basis(op::KLocalOperator) = op.hilb

operators(op::KLocalOperator) = (op,)

densemat(op::KLocalOperator) = op.mat

conn_type(top::Type{KLocalOperator{H,SV,M,Vel,Vti,Vtc,Vtv,OC}}) where {H, SV, M, Vel, Vti, Vtc, Vtv, OC} =
    OpConnection{Vel, eltype(Vtc), eltype(Vtv)}
conn_type(op::KLocalOperator{H,SV,M,Vel,Vti,Vtc,Vtv,OC}) where {H, SV, M, Vel, Vti, Vtc, Vtv, OC} =
    OpConnection{Vel, eltype(Vtc), eltype(Vtv)}

# Copy
function duplicate(op::KLocalOperator)
    KLocalOperator(basis(op), deepcopy(op.sites), deepcopy(op.hilb_dims), deepcopy(op.mat),
                    deepcopy(op.mel), deepcopy(op.to_change),
                    deepcopy(op.new_values), deepcopy(op.op_conns),
                    deepcopy(op.new_indices))
end

##
function row_valdiff!(conn::OpConnection, op::KLocalOperator, v::AState)
    # Find row index
    r = local_index(v, basis(op), sites(op))

    append!(conn, op.op_conns[r])
end

function row_valdiff_index!(conn::OpConnectionIndex, op::KLocalOperator, v::AState)
    # Find row index
    r = local_index(v, basis(op), sites(op))

    # Return values
    mel = op.mel[r]
    ids = op.new_indices[r]

    append!(conn, (mel, ids))
end

function map_connections(fun::Function, op::KLocalOperator, v::AState)
    r = local_index(v, basis(op), sites(op))
    for (mel, changes) = op.op_conns[r]
        fun(mel, changes, v)
    end
    return nothing
end

function accumulate_connections!(acc::AbstractAccumulator, op::KLocalOperator, v::AState)
    # If it is a doublestate, we are probably computing Operator x densitymatrix,
    # so we only iterate along the column of v
    if v isa ADoubleState
        r = local_index(col(v), basis(op), sites(op))
    else
        r = local_index(v, basis(op), sites(op))
    end

    for (mel,changes)=op.op_conns[r]
        acc(mel, changes, v)
    end

    return acc
end

Base.:-(op::KLocalOperator) = KLocalOperator(op, -op.mat)

_add_samesite(op_l::KLocalOperator, op_r::KLocalOperator) = _add_samesite!(duplicate(op_l), op_r)

function _add_samesite!(op_l::KLocalOperator, op_r::KLocalOperator)
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

Base.:*(a::Number, b::KLocalOperator) =
    _op_alpha_prod(b,a)
_op_alpha_prod(op::KLocalOperator, a::Number) =
    KLocalOperator(op, a*op.mat)

function Base.:*(opl::KLocalOperator, opr::KLocalOperator)
    @assert shape(basis(opl)) == shape(basis(opr))

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

        # inverted also in QuantumOptics... who knows why
        mat = kron(opr.mat, opl.mat)
        return KLocalOperatorRow(basis(opl), [sl, sr], mat)
    else
        sites_new = sort(vcat(sites(opl), sites(opr)))
        ids_l = [findfirst(i .==sites_new) for i=sites(opl)]
        ids_r = [findfirst(i .==sites_new) for i=sites(opr)]
        throw("Tensor product between two operators on disjoint support but on
        more than 1 site each must still be implemneted. Sites in question are $(sites(opl)) and $(sites(opr)).")
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

        # inverted also in QuantumOptics... who knows why
        mat_r = kron(reverse(matrices)...)
        prod_mat = reversed ? mat_r*opl.mat : opl.mat*mat_r

        return KLocalOperator(opl, prod_mat)
    else
        sites_new = sort(vcat(sites(opl), sites(opr)))
        ids_l = [findfirst(i .==sites_new) for i=sites(opl)]
        ids_r = [findfirst(i .==sites_new) for i=sites(opr)]
        throw("Tensor product between two operators on overlapping support but
        both on more than 1 site must still be implemneted.")
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
Base.eltype(T::Type{KLocalOperator{H,SV,M,Vel,Vti,Vtc,Vtv,OC}}) where {H,SV,M,Vel,Vti,Vtc,Vtv,OC} =
    eltype(eltype(Vel))

Base.show(io::IO, op::KLocalOperator) = begin
    T    = eltype(op)
    s    = sites(op)
    h    = basis(op)
    dims = hilb_dims(op)
    mat  = densemat(op)

    print(io, "KLocalOperatorRow($T, $h, $s, $dims, $mat)")
end

Base.show(io::IO, m::MIME"text/plain", op::KLocalOperator) = begin
    T    = eltype(op)
    s    = sites(op)
    h    = basis(op)
    dims = hilb_dims(op)
    mat  = densemat(op)

    print(io, "KLocalOperator($T)\n  Hilb: $h\n  sites: $s  (size: $dims)\n")
    Base.print_array(io, mat)
end
