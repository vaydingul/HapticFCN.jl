using Distributed
# @everywhere used for the implementation of the modules across multiple workers
# Without it, the code does not run

# Add the modules to the ´read´ path of the Julia
@everywhere push!(LOAD_PATH, "/scratch/users/vaydingul20/workfolder/COMP541_Project/src/modules/")
# Custom modules
@everywhere using TUM69: load_accel_data, load_image_data
@everywhere using Preprocess: process_accel_signal, process_image, augment_image
@everywhere using Utils: kfold, notify!, a_type
# Ready-to-use modules 
using JLD2, Random
using CUDA

notify!("Script started! -- hn")

# Set path
path = CUDA.functional() ? "/userfiles/vaydingul20/data/new" : "data/new" 
DATA_PATH = isdir(path) && path

notify!("Data reading started! -- hn")
# Load data
X_train, y_train, _, _, material_dict = load_accel_data(DATA_PATH; mode = "normal")

notify!("Preprocessing started! -- hn")
# Apply preprocessing on the accelaration data
X_train, y_train = process_accel_signal(X_train, y_train)
# Seperate into 3 folds for training
notify!("10-FOLD started!")
kf = kfold(X_train, y_train; fold = 10, minibatch_size = 100, atype = a_type(Float32))
notify!("10-FOLD finished!")
JLD2.@save "10fold.jld2" kf = kf
