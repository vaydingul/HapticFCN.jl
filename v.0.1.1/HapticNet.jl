#=
using Pkg; 
packages = ["Knet", "AutoGrad", "Random", "Test", "MLDatasets", "CUDA", "Plots", "GR","Statistics",
            "IterTools", "StatsBase", "DSP", "Images", "DelimitedFiles", "MultivariateStats", "PyPlot", "PyCall"];
Pkg.add(packages);
=#
include("../src/modules/TUM69.jl")
include("../src/modules/Preprocess.jl")
include("../src/modules/Network.jl")
include("../src/modules/Utils.jl")

## Third party packages
using Knet: KnetArray, adam, relu, minibatch
using AutoGrad, Knet, CUDA, JLD2


## Handwritten modules
using .TUM69: load_accel_data   # Data reading
using .Preprocess: process_accel_signal # Preprocessing on the data
using .Network: GCN, train_summarize!, accuracy4, nll4, GenericMLP # Construction of custom network
using .Utils: notify

notify("Script started!")
AutoGrad.set_gc_function(AutoGrad.default_gc)

# Array type setting for GPU usage
a_type() = (CUDA.functional() ? KnetArray{Float32} : Array{Float32})

# CUDA information
GC.gc(true)
CUDA.device()

notify("Data reading started!")
path = CUDA.functional() ? "/userfiles/vaydingul20/data/new" : "./../data/new" # path of the main data
DATA_PATH = isdir(path) && path

X_train, y_train,
X_test, y_test, 
material_dict = @time load_accel_data(DATA_PATH; mode = "normal");  # Data loading routine
notify("Data reading is done!")



println("X_train = ", summary(X_train))
println("y_train = ", summary(y_train))
println("X_test  = ", summary(X_test))
println("y_test  = ", summary(y_test))
println("material_dict = ", summary(material_dict))

notify("Preprocessing started!")
#Preprocessing on the acceleration data
@time X_train_modified, y_train_modified = process_accel_signal(X_train, y_train);
@time X_test_modified, y_test_modified = process_accel_signal(X_test, y_test);
notify("Preprocessing is done!")


println("X_train = ", summary(X_train_modified))
println("y_train = ", summary(y_train_modified))
println("X_test  = ", summary(X_test_modified))
println("y_test  = ", summary(y_test_modified))
println("material_dict = ", summary(material_dict))



# Some constants that will be used in the network model
MINIBATCH_SIZE = 10
INPUT_SIZE = size(X_test_modified)[1:3]
OUTPUT_SIZE = size(collect(keys(material_dict)))[1];

# Minibatching
dtrn = minibatch(X_train_modified, y_train_modified, MINIBATCH_SIZE; xtype = a_type(), shuffle = true)
dtst = minibatch(X_test_modified, y_test_modified, MINIBATCH_SIZE; xtype = a_type(), shuffle = true);
notify("Minibatching is done!")
#=
model = GCN(INPUT_SIZE, OUTPUT_SIZE, 
       [(25, 150, 1 , relu, 0.0, (1, 1), (1, 1), (2, 2), false)]; 
    hidden=[10], optimizer_type = adam, lr = 1e-4, loss_fnc=nll, accuracy_fnc = accuracy, atype=a_type())
=#

model = GCN(INPUT_SIZE, OUTPUT_SIZE, 
       [(3, 3, 50 , relu, 0.0, (1, 1), (1, 1), (2, 2), true),
        (3, 3, 100, relu, 0.0, (1, 1), (1, 1), (2, 2), false),
        (3, 3, 150, relu, 0.0, (1, 1), (1, 1), (2, 2), false),
        (3, 3, 200, relu, 0.0, (1, 1), (1, 1), (2, 2), false),
        (4, 12,400, relu, 0.5, (1, 0), (1, 1), (1, 1), false),
        (1, 1, 250, relu, 0.5, (0, 0), (1, 1), (1, 1), false),
        (1, 1, OUTPUT_SIZE , relu, 0.5, (0, 0), (1, 1), (1, 1), false),
        ]; 
    hidden=[], optimizer_type = adam, lr = 1e-4, loss_fnc=nll4, accuracy_fnc = accuracy4, atype=a_type())



# Training routine
# Currently, the model is not working due to the issue mentioned in: https://github.com/denizyuret/Knet.jl/issues/624#
# As soon as it is solved, I hope the model will be accurately working.

notify("Ttraining started!")
res = train_summarize!(model, dtrn, dtst; 
                    train_type = "epoch", progress_bar = false ,fig = false, info = true, 
                    epoch = 1000, conv_epoch = 50, max_conv_cycle = 20)

lval = model(dtrn)
notify("Training is done!")
notify("Loss = $lval")

JLD2.@save "final.jld2" model res

