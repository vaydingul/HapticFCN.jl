#include("../src/modules/TUM69.jl")
#include("../src/modules/Preprocess.jl")
#include("../src/modules/Network.jl")
#include("../src/modules/Utils.jl")
#include("../src/modules/Model.jl")
#include("../src/modules/Metrics.jl")

## Third party packages
using Knet: KnetArray, adam, relu, minibatch
using AutoGrad, Knet, CUDA, JLD2, Test


## Handwritten modules
using .TUM69: load_accel_data, load_image_data   # Data reading
using .Preprocess: process_accel_signal, process_image, augment_image # Preprocessing on the data
using .Network: GCN, nll4, accuracy4, train_summarize!
using .Utils: notify, a_type
using .Model: HapticNet, VisualNet
using .Metrics: visualize, confusion_matrix

MINIBATCH_SIZE = 2;
PATH = "../data/trial"

X_train, y_train, X_test, y_test, material_dict = @time load_image_data(PATH; mode = "baseline")

@time X_train, y_train = augment_image(X_train, y_train);

@time X_train_modified, y_train_modified = process_image(X_train, y_train);
@time X_test_modified, y_test_modified = process_image(X_test, y_test);



vn = VisualNet(; atype = a_type(Float32))
dtrn = minibatch(X_train_modified, y_train_modified, MINIBATCH_SIZE; xtype = a_type(Float32), shuffle = true)
dtst = minibatch(X_test_modified, y_test_modified, MINIBATCH_SIZE; xtype = a_type(Float32), shuffle = true);


res = train_summarize!(vn.model, dtrn, dtst; 
                    train_type = "epoch", progress_bar = true ,fig = true, info = true, 
                    epoch = 10, conv_epoch = 50, max_conv_cycle = 20)

