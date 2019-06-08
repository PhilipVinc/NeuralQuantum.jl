abstract type AbsLinearOperator end

"""
    row_valdiff(op::AbsLinearOperator, v::State) -> OpConnection

Returns all non-zero elements in the row represented by the state `v` of
operator `op`. The result is an `OpConnection` type.
"""
row_valdiff(op::AbsLinearOperator, v::State) = row_valdiff!(OpConnection(op), op, v)

"""
    row_valdiff!(opconn::OpConnection, op::AbsLinearOperator, v::State)

Returns all non-zero elements in the row represented by the state `v` of
operator `op` by mutating in-place the `OpConnection` object `opconn`
"""
function row_valdiff! end

row_valdiff_index(op::AbsLinearOperator, v::State) = row_valdiff_index!(OpConnectionIndex(op), op, v)
function row_valdiff_index! end
