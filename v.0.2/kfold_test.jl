include("../src/modules/TUM69.jl")
include("../src/modules/Preprocess.jl")
include("../src/modules/Network.jl")
include("../src/modules/Utils.jl")
include("../src/modules/Model.jl")
include("../src/modules/Metrics.jl")

## Third party packages
using Knet: KnetArray, adam, relu, minibatch
using AutoGrad, Knet, CUDA, JLD2, Test


## Handwritten modules
using .TUM69: load_accel_data, load_image_data   # Data reading
using .Preprocess: process_accel_signal, process_image, augment_image # Preprocessing on the data
using .Network: GCN, nll4, accuracy4, train_summarize!
using .Utils: notify, a_type, kfold
using .Model: HapticNet, VisualNet
using .Metrics: visualize, confusion_matrix

MINIBATCH_SIZE = 10;
PATH = "../data/trial"

X_train, y_train, X_test, y_test, material_dict = @time load_accel_data(PATH; mode = "baseline")


@time X_train_modified, y_train_modified = process_accel_signal(X_train, y_train);
@time X_test_modified, y_test_modified = process_accel_signal(X_test, y_test);

kf = kfold(X_train_modified, y_test_modified;m atype = a_type(Float32))