push!(LOAD_PATH, "/scratch/users/vaydingul20/workfolder/COMP541_Project/src/modules/")
using Distributed

@everywhere
    using Pkg; Pkg.activate(".")  # required
    using Model: HapticNet, VisualNet, train_epoch!, save_as_jld2
    using TUM69: load_accel_data, load_image_data
    using Preprocess: process_accel_signal, process_image, augment_image
    using Utils: kfold, notify!, a_type
end


using JLD2, Random
notify!("Script started!")
path = CUDA.functional() ? "/userfiles/vaydingul20/data/new" : "data/new" # path of the main data
DATA_PATH = isdir(path) && path
alexnet_PATH = "/scratch/users/vaydingul20/workfolder/COMP541_Project/alexnet.mat"

notify!("Data reading started!")

X_train, y_train, _, _, material_dict = load_image_data(DATA_PATH; mode = "normal")

notify!("Preprocessing started!")

X_train, y_train = augment_image(X_train, y_train)
X_train, y_train = process_image(X_train, y_train)
#X_test, y_test = process_accel_signal(X_test, y_test)

kf = kfold(X_train, y_train, fold = 3, atype = a_type(Float32))
results = []

notify!("Training started!")

for (ix, dtrn, dtst) in enumerate(kf.folds)
    
    notify!("Training $ix started!")

    vn = VisualNet(alexnet_PATH; atype = a_type(Float32))
    res = train_epoch!(vn, dtrn, dtst; progress_bar = false, fig = false, info = true, epoch = 3000)
    save_as_jld2(vn, "vn-$ix.jld2")
    push!(results, res)
    
end

JLD2.@save "results_vn.jld2" results = results
