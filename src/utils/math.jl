# batched
"""
    sum_autobatch(v)

it's equivalent to sum(v, dims=1) except for the case
of a vector, where it will return a scalar instead of
a 1-element vector.
"""
sum_autobatch(v) = sum(v, dims=1)
sum_autobatch(v::Vector) = sum(v)


"""
    _batched_outer_prod!(R, vb, wb)

Efficiently performs the outer product R[i] .= vb[i] .* wb[i]'
along the batch dimension i, assuming that the batch dimension
is the last.
Internally uses the fact that R is a StridedView
"""
@inline function _batched_outer_prod!(R::StridedView, vb, wb)
    #@unsafe_strided R begin
        @inbounds @simd for i=1:size(R, 3)
            for j=1:size(wb, 1)
                for k=1:size(vb, 1)
                    R[k,j,i] = vb[k,i]*conj(wb[j,i])
                end
            end
        end
    #end

    #=@unsafe_strided R vb wb begin
        for i=1:size(R, 3)
            BLAS.ger!(1.0, vb[:,i], wb[:,i], R[:,:,i])
        end
    end=#
    return R
end
