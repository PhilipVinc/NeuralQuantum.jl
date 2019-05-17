export load_from_dict!

"""
    load_from_dict!(net, dict)

Sets the weights of `net` to those stored in `dict`
"""
function load_from_dict!(net, dict)
    for key=keys(dict)
        f = getfield(net, key)
        f .= dict[key]
    end
    return net
end
