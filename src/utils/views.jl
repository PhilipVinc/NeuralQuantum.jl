function _uview_destride(vec::Strided.StridedView{T,3}, i) where {T}
    stride_last = vec.strides[end]

    start = vec.offset + (i-1) * stride_last
    len   = prod(vec.size[1:end-1])

    return reshape(uview(vec.parent, start:start+len), vec.size[1], vec.size[2])
end

function _uview_vec_destride(vec::Strided.StridedView{T,3}, i) where {T}
    stride_last = vec.strides[end]

    start = vec.offset + (i-1) * stride_last
    len   = prod(vec.size[1:end-1])

    return uview(vec.parent, start+1:start+len)
end
