export train_epoch!, train_generic!, save_as_jld2

include("network_ops.jl");
include("vn.jl")
include("hn.jl")

#using JLD2
#import train_epoch!

# Extension of ´train_epoch!´ function for VisualNet and HapticNet
train_epoch!(vn::VisualNet, dtrn, dtst; o...) = train_epoch!(vn.model, dtrn, dtst; o...)
train_epoch!(hn::HapticNet, dtrn, dtst; o...) = train_epoch!(hn.model, dtrn, dtst; o...)
train_generic!(vn::VisualNet, dtrn, dtst; o...) = train_generic!(vn.model, dtrn, dtst; o...)
train_generic!(hn::HapticNet, dtrn, dtst; o...) = train_generic!(hn.model, dtrn, dtst; o...)
# Basic tool to save models to JLD2 file
save_as_jld2(vn::VisualNet, fname) = JLD2.@save fname model = vn.model
save_as_jld2(hn::HapticNet, fname) = JLD2.@save fname model = hn.model

