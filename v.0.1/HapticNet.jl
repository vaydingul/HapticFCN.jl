using Pkg; 
packages = ["Knet", "AutoGrad", "Random", "Test", "MLDatasets", "CUDA", "Plots", "GR","Statistics",
            "IterTools", "StatsBase", "DSP", "Images", "DelimitedFiles", "MultivariateStats", "PyPlot", "PyCall"];
Pkg.add(packages);

include("../src/modules/TUM69.jl")
include("../src/modules/Preprocess.jl")
include("../src/modules/Network.jl")

## Third party packages
using Knet: KnetArray, adam, relu, minibatch
using CUDA: CuArray
import CUDA
using AutoGrad

## Handwritten modules
using .TUM69: load_accel_data   # Data reading
using .Preprocess: process_accel_signal # Preprocessing on the data
using .Network: GeneriCONV, train_summarize!, accuracy4 # Construction of custom network


# Trick from Deniz Hoca to deal with this issue: https://github.com/denizyuret/Knet.jl/issues/524
using Knet
function Knet.KnetArray(x::CuArray{T,N}) where {T,N}
    p = Base.bitcast(Knet.Cptr, pointer(x))
    k = Knet.KnetPtr(p, sizeof(x), Int(CUDA.device().handle), x)
    KnetArray{T,N}(k, size(x))
end

# Array type setting for GPU usage
a_type() = (CUDA.functional() ? KnetArray{Float32} : Array{Float32})

# CUDA information
GC.gc(true)
CUDA.device()

path = "/userfiles/vaydingul20/data/new/" # path of the main data
DATA_PATH = isdir(path) && path

X_train, y_train,
X_test, y_test, 
material_dict = @time load_accel_data(DATA_PATH; mode = "normal");  # Data loading routine

println("X_train = ", summary(X_train))
println("y_train = ", summary(y_train))
println("X_test  = ", summary(X_test))
println("y_test  = ", summary(y_test))
println("material_dict = ", summary(material_dict))

#Preprocessing on the acceleration data
@time X_train_modified, y_train_modified = process_accel_signal(X_train, y_train);
@time X_test_modified, y_test_modified = process_accel_signal(X_test, y_test);

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
dtrn = minibatch(X_train_modified, y_train_modified, MINIBATCH_SIZE; xtype = a_type())
dtst = minibatch(X_test_modified, y_test_modified, MINIBATCH_SIZE; xtype = a_type());

# Generic model construction routine
hn = GeneriCONV(INPUT_SIZE, 0.0, [(3, 3, 50, true), (3, 3, 100, true), (3, 3, 150, true),
            (3, 3, 200, true), (3, 12, 400, false), (1, 1, 250, false), (1, 1, OUTPUT_SIZE, false)];
            hidden = [], f = relu, a_type = a_type(), pdrop = 0.5, 
            optimizer_type = adam, lr = 1e-4);

# Training routine
# Currently, the model is not working due to the issue mentioned in: https://github.com/denizyuret/Knet.jl/issues/624#
# As soon as it is solved, I hope the model will be accurately working.
res = train_summarize!(hn, dtrn, dtst; 
                       train_type = "epoch", fig = true, info = true, 
                       epoch = 100, conv_epoch = 50, max_conv_cycle = 20)

J = @diff hn(dtrn)


