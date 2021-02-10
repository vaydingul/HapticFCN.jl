export HapticNet


struct HapticNet
    #=

        Ready to use wrapper for the HapticNet

    =#

    model::GCN # GCN model that it wraps
    
end

function HapticNet(s::String)
    #=

        Constructor definition for pretrained networks

    =#
    fname = endswith(s, ".jld2") && s
    JLD2.@load fname model
    HapticNet(model)

end

function HapticNet(; i = (50, 300, 1), o = 69, lrn = true, atype = Array{Float32})
    #=

        Constructor definition for default HapticNet

    =#

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


