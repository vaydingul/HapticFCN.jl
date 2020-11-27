#=
using Pkg; 
packages = ["Knet", "AutoGrad", "Random", "Test", "MLDatasets", "CUDA", "Plots", "GR","Statistics",
            "IterTools", "StatsBase", "DSP", "Images", "DelimitedFiles", "MultivariateStats", "PyPlot", "PyCall"];
Pkg.add(packages);
=#
include("../src/modules/TUM69.jl")
include("../src/modules/Preprocess.jl")
include("../src/modules/Network.jl")

## Third party packages
using Knet: KnetArray, adam, relu, minibatch, save, sgd
import CUDA

## Handwritten modules
using .TUM69: load_accel_data   # Data reading
using .Preprocess: process_accel_signal # Preprocessing on the data
using .Network: GeneriCONV, train_summarize!, accuracy4 # Construction of custom network
notify(str) = run(`curl https://notify.run/fnx04zT7QmOlLLa6 -d $str`)

notify("SCRIPT STARTED!!")

# Array type setting for GPU usage
a_type() = (CUDA.functional() ? KnetArray{Float32} : Array{Float32})
# CUDA information
CUDA.device()

path = "/userfiles/vaydingul20/data/new/"
DATA_PATH = isdir(path) && path

X_train, y_train,
X_test, y_test, 
material_dict = load_accel_data(DATA_PATH; mode = "normal"); 
notify("DATA READING DONE!!")

println("X_train = ", summary(X_train))
println("y_train = ", summary(y_train))
println("X_test  = ", summary(X_test))
println("y_test  = ", summary(y_test))
println("material_dict = ", summary(material_dict))

@time X_train_modified, y_train_modified = process_accel_signal(X_train, y_train);
@time X_test_modified, y_test_modified = process_accel_signal(X_test, y_test);
notify("DATA PREPROS DONE!!")

println("X_train = ", summary(X_train_modified))
println("y_train = ", summary(y_train_modified))
println("X_test  = ", summary(X_test_modified))
println("y_test  = ", summary(y_test_modified))
println("material_dict = ", summary(material_dict))

@show MINIBATCH_SIZE = 10;
@show INPUT_SIZE = size(X_test_modified)[1:3]
@show OUTPUT_SIZE = size(collect(keys(material_dict)))[1]

dtrn = minibatch(X_train_modified, y_train_modified, MINIBATCH_SIZE; xtype = a_type())
dtst = minibatch(X_test_modified, y_test_modified, MINIBATCH_SIZE; xtype = a_type())
notify("MINIBATCHING DONE!!")


hn = GeneriCONV(INPUT_SIZE, 0.0, [(3, 3, 50, true), (3, 3, 100, true), (3, 3, 150, true),
            (3, 3, 200, true), (3, 12, 400, false), (1, 1, 250, false), (1, 1, OUTPUT_SIZE, false)];
            hidden = [], f = relu, a_type = a_type(), pdrop = 0.5, 
            optimizer_type = adam, lr = 1e-4)


for k in 1:100
    notify("TRAINING STARTED!!")

    res = train_summarize!(hn, dtrn, dtst; 
                       train_type = "epoch", fig = false, info = true, 
                       epoch = 10, conv_epoch = 50, max_conv_cycle = 20)

    mod(k, 10) == 0 ? save("$k.jld2", "model", hn, "result", res) : nothing

end

notify("TRAINING FINISHED!!")
