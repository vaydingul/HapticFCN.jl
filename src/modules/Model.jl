module Model


include("model//vn.jl"); export VisualNet
include("model//hn.jl"); export HapticNet
include("model//model_ops.jl"); export train_epoch!, save_as_jld2
    
end