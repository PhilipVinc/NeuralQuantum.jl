using NeuralQuantum
using LinearAlgebra, SparseArrays
using Test

@testset "KLocalOperator" begin
mat1 = [0.0 0.0; 0.1 0.0]
mat2 = [0.5 0.1; 0.1 0.0]
matsum = mat1 + mat2

sts = [1]
hilb_dims = [2]

op = KLocalOperatorRow(sts, hilb_dims, mat1)
op2 = KLocalOperatorRow(sts, hilb_dims, mat2)

ops = KLocalOperatorSum(op)
ops2 = ops + op2
@test length(NeuralQuantum.operators(ops)) == 1
@test first(NeuralQuantum.operators(ops2)).mat == matsum

# Check that index works for a single site
check = sparse(zeros(size(matsum)))
v = NAryState(2, 1)
for i=1:spacedimension(v)
    set_index!(v, i)
    diffs = row_valdiff_index(ops2, v)
    for (mel,j)=diffs
        check[i,j] += mel
    end
end
@test check == matsum

# check that the reconstructed matrix is fine
check = sparse(zeros(size(matsum)))
v = NAryState(Float64, 2, 1)
for i=1:spacedimension(v)
    set_index!(v, i)
    for (mel, changes) = row_valdiff(ops2, v)
        set_index!(v, i)
        for (id,val) = changes
            setat!(v, id, val)
        end
        j = index(v)
        check[i,j] += mel
    end
end
@test check == matsum

# check that by kroneckering times identity it's fine
ms = kron(Matrix(I, 2, 2), matsum)
#ms = kron(matsum, Matrix(I, 2, 2))
check = sparse(zeros(size(ms)))
v = NAryState(Float64, 2, 2)
for i=1:spacedimension(v)
    set_index!(v, i)
    for (mel, changes) = row_valdiff(ops2, v)
        set_index!(v, i)
        for (id,val)=changes
            setat!(v, id, val)
        end
        j = index(v)
        check[i,j] += mel
    end
end
@test check == ms

mat12 = kron(mat1, mat2)
op12 = KLocalOperatorRow([1,2], [2,2], mat12)
check = sparse(zeros(size(mat12)))
v = NAryState(Float64, 2, 2)
for i=1:spacedimension(v)
    set_index!(v, i)
    for (mel, changes) = row_valdiff(op12, v)
        set_index!(v, i)
        for (id,val)=changes
            setat!(v, id, val)
        end
        j = index(v)
        check[i,j] += mel
    end
end
@test check == mat12

op3 = KLocalOperatorRow([2], hilb_dims, mat2)
m3 = kron(mat2, Matrix(I, 2, 2))
ms = kron(Matrix(I, 2, 2), matsum)

mtot = m3 #ms + m3
opstot = op3 #ops2 + op3

check = sparse(zeros(size(mtot)))
v = NAryState(Float64, 2, 2)
for i=1:spacedimension(v)
    set_index!(v, i)
    for (mel, changes) = row_valdiff(opstot, v)
        set_index!(v, i)
        for (id,val)=changes
            setat!(v, id, val)
        end
        j = index(v)
        check[i,j] += mel
    end
end
@test check == m3

end
