push!(LOAD_PATH, "/scratch/users/vaydingul20/workfolder/COMP541_Project/src/modules/")
using Distributed

@everywhere using Pkg; Pkg.activate(".")  # required
@everywhere using Model: HapticNet, VisualNet, train_epoch!, save_as_jld2
@everywhere using TUM69: load_accel_data, load_image_data
@everywhere using Preprocess: process_accel_signal, process_image, augment_image
@everywhere using Utils: kfold, notify!, a_type

using JLD2, Random

notify!("Script started!")


path = CUDA.functional() ? "/userfiles/vaydingul20/data/new" : "data/new" # path of the main data
DATA_PATH = isdir(path) && path

notify!("Data reading started!")

X_train, y_train, _, _, material_dict = load_accel_data(DATA_PATH; mode = "normal")

notify!("Preprocessing started!")

X_train, y_train = process_accel_signal(X_train, y_train)
#X_test, y_test = process_accel_signal(X_test, y_test)

kf = kfold(X_train, y_train, fold = 3, atype = a_type(Float32))
results = []

notify!("Training started!")

for (ix, dtrn, dtst) in enumerate(kf.folds)

    notify!("Training $ix started!")

    hn = HapticNet(; atype = a_type(Float32))
    res = train_epoch!(hn, dtrn, dtst; progress_bar = false, fig = false, info = true, epoch = 3000)
    save_as_jld2(hn, "hn-$ix.jld2")
    push!(results, res)
    
end

JLD2.@save "results_hn.jld2" results = results
