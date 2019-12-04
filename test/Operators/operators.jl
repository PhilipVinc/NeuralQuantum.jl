using NeuralQuantum
using NeuralQuantum: KLocalOperatorSum, row_valdiff_index, local_dim
using LinearAlgebra, SparseArrays
using Test

@testset "KLocalOperator" begin
hilb = HomogeneousSpin(2)

mat1 = [0.0 0.0; 0.1 0.0]
mat2 = [0.5 0.1; 0.1 0.0]
matsum = mat1 + mat2

sts = [1]
hilb_dims = [2]

op  = KLocalOperatorRow(hilb, sts, mat1)
op2 = KLocalOperatorRow(hilb, sts, mat2)

ops = KLocalOperatorSum(op)
ops2 = ops + op2
@test length(NeuralQuantum.operators(ops)) == 1
@test first(NeuralQuantum.operators(ops2)).mat == matsum

# Check that index works for a single site
check = sparse(zeros(size(matsum)))
v = state(hilb)
for i=1:local_dim(hilb)
    set!(v, hilb, i)
    diffs = row_valdiff_index(ops2, v)
    for (mel,j)=diffs
        check[i,j] += mel
    end
end
@test check == matsum

# check that the reconstructed matrix is fine
check = sparse(zeros(size(matsum)))
v = state(hilb)
for i=1:local_dim(hilb)
    set_index!(v, hilb, i)
    for (mel, changes) = row_valdiff(ops2, v)
        set!(v, hilb, i)
        for (id,val) = changes
            setat!(v, hilb, id, val)
        end
        j = index(hilb, v)
        check[i,j] += mel
    end
end
@test check == matsum

# check that by kroneckering times identity it's fine
ms = kron(Matrix(I, 2, 2), matsum)
#ms = kron(matsum, Matrix(I, 2, 2))
check = sparse(zeros(size(ms)))
v = state(hilb)
for i=1:spacedimension(hilb)
    set_index!(v, hilb, i)
    for (mel, changes) = row_valdiff(ops2, v)
        set!(v, hilb, i)
        for (id,val)=changes
            setat!(v, hilb, id, val)
        end
        j = index(hilb, v)
        check[i,j] += mel
    end
end
@test check == ms

mat12 = kron(mat1, mat2)
op12 = KLocalOperatorRow(hilb, [1,2], mat12)
check = sparse(zeros(size(mat12)))
v = state(hilb)
for i=1:spacedimension(hilb)
    set_index!(v, hilb, i)
    for (mel, changes) = row_valdiff(op12, v)
        set_index!(v, hilb, i)
        for (id,val)=changes
            setat!(v, hilb, id, val)
        end
        j = index(hilb, v)
        check[i,j] += mel
    end
end
@test check == mat12

op3 = KLocalOperatorRow(hilb, [2], mat2)
m3 = kron(mat2, Matrix(I, 2, 2))
ms = kron(Matrix(I, 2, 2), matsum)

mtot = m3 #ms + m3
opstot = op3 #ops2 + op3

check = sparse(zeros(size(mtot)))
v = state(hilb)
for i=1:spacedimension(hilb)
    set_index!(v, hilb, i)
    for (mel, changes) = row_valdiff(opstot, v)
        set!(v, hilb, i)
        for (id,val)=changes
            setat!(v, hilb, id, val)
        end
        j = index(hilb, v)
        check[i,j] += mel
    end
end
@test check == m3

end
