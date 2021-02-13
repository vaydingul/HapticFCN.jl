module HapticFCN

include("./modules/TUM69.jl"); 
include("./modules/Network.jl"); 
include("./modules/Utils.jl"); 

using HapticFCN.TUM69
using HapticFCN.Network
using HapticFCN.Utils


export HapticData, VisualData, HapticNet, VisualNet, train_epoch!, save_as_jld2, a_type, notify!


end # module
