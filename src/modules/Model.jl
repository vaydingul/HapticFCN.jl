module Model
include("Network.jl")

using .Network
using MAT, Knet, JLD2

export VisualNet, HapticNet

struct HapticNet

    model#::GCN

end

function HapticNet(s::String)

    fname = endswith(s, ".jld2") && s
    JLD2.@load fname model
    HapticNet(model)

end

function HapticNet(; i = (50, 300, 1), o = 69, lrn = true, atype = Array{Float32})
    
    model = GCN(i, o, 
       [(3, 3, 50, relu, 0.0, (1, 1), (1, 1),  (2, 2),(2, 2), lrn),
        (3, 3, 100, relu, 0.0, (1, 1), (1, 1), (2, 2),(2, 2), false),
        (3, 3, 150, relu, 0.0, (1, 1), (1, 1), (2, 2),(2, 2), false),
        (3, 3, 200, relu, 0.0, (1, 1), (1, 1), (2, 2),(2, 2), false),
        (4, 12, 400, relu, 0.5, (1, 0), (1, 1), (1, 1),(1, 1), false),
        (1, 1, 250, relu, 0.5, (0, 0), (1, 1), (1, 1),(1, 1), false),
        (1, 1, o, relu, 0.5, (0, 0), (1, 1), (1, 1),(1, 1), false),
        ]; 
    hidden=[], optimizer_type=adam, lr=1e-4, loss_fnc=nll4, accuracy_fnc=accuracy4, atype=atype)

    return HapticNet(model)
end

struct VisualNet

    model#::GCN

end

function VisualNet(s::String)

    fname = endswith(s, ".jld2") && s
    JLD2.@load fname model
    VisualNet(model)

end

function VisualNet(; i = (384, 384, 3), o = 69, lrn = true, atype = Array{Float32})

    alexnet = matread("alexnet.mat")
    conv1w = alexnet["params"]["value"][1]; conv1w = conv1w[:, :, :, 1:Int(size(conv1w,4) / 2)];
    conv1b = alexnet["params"]["value"][2]; conv1b = conv1b[1:Int(size(conv1b,1) / 2)];
    conv2w = alexnet["params"]["value"][3];
    conv2b = alexnet["params"]["value"][4];
    conv3w = alexnet["params"]["value"][5]; conv3w = conv3w[:, :, :, 1:Int(size(conv3w,4) / 2)];
    conv3b = alexnet["params"]["value"][6]; conv3b = conv3b[1:Int(size(conv3b,1) / 2)];
    conv4w = alexnet["params"]["value"][7]; conv4w = conv4w[:, :, :, 1:Int(size(conv4w,4) / 2)];
    conv4b = alexnet["params"]["value"][8]; conv4b = conv4b[1:Int(size(conv4b,1) / 2)];
    conv5w = alexnet["params"]["value"][9]; 
    conv5b = alexnet["params"]["value"][10];

    conv1_stride = Tuple(convert(Array{Int}, vec(alexnet["layers"]["block"][1]["stride"])))
    conv2_stride = Tuple(convert(Array{Int}, vec(alexnet["layers"]["block"][4]["stride"])))
    conv3_stride = Tuple(convert(Array{Int}, vec(alexnet["layers"]["block"][7]["stride"])))
    conv4_stride = Tuple(convert(Array{Int}, vec(alexnet["layers"]["block"][9]["stride"])))
    conv5_stride = Tuple(convert(Array{Int}, vec(alexnet["layers"]["block"][11]["stride"])))

    conv1_pad = Tuple(convert(Array{Int}, vec(alexnet["layers"]["block"][1]["pad"])[1:2]))
    conv2_pad = Tuple(convert(Array{Int}, vec(alexnet["layers"]["block"][4]["pad"])[1:2]))
    conv3_pad = Tuple(convert(Array{Int}, vec(alexnet["layers"]["block"][7]["pad"])[1:2]))
    conv4_pad = Tuple(convert(Array{Int}, vec(alexnet["layers"]["block"][9]["pad"])[1:2]))
    conv5_pad = Tuple(convert(Array{Int}, vec(alexnet["layers"]["block"][11]["pad"])[1:2]))

    conv1_pool_window = Tuple(convert(Array{Int}, vec(alexnet["layers"]["block"][3]["poolSize"])))
    conv2_pool_window = Tuple(convert(Array{Int}, vec(alexnet["layers"]["block"][6]["poolSize"])))
    conv3_pool_window = Tuple([1, 1])#Tuple(alexnet["layers"]["block"][7]["pad"])
    conv4_pool_window = Tuple([1, 1])#Tuple(alexnet["layers"]["block"][9]["pad"])
    conv5_pool_window = Tuple(convert(Array{Int}, vec(alexnet["layers"]["block"][13]["poolSize"])))

    conv1_pool_stride = Tuple(convert(Array{Int}, vec(alexnet["layers"]["block"][3]["stride"])))
    conv2_pool_stride = Tuple(convert(Array{Int}, vec(alexnet["layers"]["block"][6]["stride"])))
    conv3_pool_stride = Tuple([1, 1])#Tuple(alexnet["layers"]["block"][7]["pad"])
    conv4_pool_stride = Tuple([1, 1])#Tuple(alexnet["layers"]["block"][9]["pad"])
    conv5_pool_stride = Tuple(convert(Array{Int}, vec(alexnet["layers"]["block"][13]["stride"])))

    model = GCN(i, o, 
       [(conv1w, conv1b , relu, 0.0, conv1_pad, conv1_stride, conv1_pool_window, conv1_pool_stride, lrn),
        (conv2w, conv2b , relu, 0.0, conv2_pad, conv2_stride, conv2_pool_window, conv2_pool_stride, lrn),
        (conv3w, conv3b , relu, 0.0, conv3_pad, conv3_stride, conv3_pool_window, conv3_pool_stride, false),
        (conv4w, conv4b , relu, 0.0, conv4_pad, conv4_stride, conv4_pool_window, conv4_pool_stride, false),
        (conv5w, conv5b , relu, 0.0, conv5_pad, conv5_stride, conv5_pool_window, conv5_pool_stride, false),
        (6, 6, 300, relu, 0.5, (0, 0), (1, 1), (1, 1), (1, 1), false),
        (1, 1, 250, relu, 0.5, (0, 0), (1, 1), (1, 1), (1, 1), false),
        (1, 1, o, relu, 0.5, (0, 0), (1, 1), (1, 1), (1, 1), false)
        ]; 
    hidden=[], optimizer_type = adam, lr = 1e-4, loss_fnc=nll4, accuracy_fnc = accuracy4, atype = atype)

    VisualNet(model)
end

    
end