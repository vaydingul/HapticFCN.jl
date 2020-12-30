using Distributed
# @everywhere used for the implementation of the modules across multiple workers
# Without it, the code does not run

# Add the modules to the ´read´ path of the Julia
@everywhere push!(LOAD_PATH, "/scratch/users/vaydingul20/workfolder/COMP541_Project/src/modules/")
# Custom modules
@everywhere using Model: HapticNet, VisualNet, train_epoch!, save_as_jld2
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
kf = kfold(X_train, y_train; fold = 3, atype = a_type(Float32))

# Initialization of the results vector that will be saved at the end of the training
results = []

notify!("Training started! -- hn")

# For each fold run training subroutine
for (ix, (dtrn, dtst)) in enumerate(kf.folds)

    notify!("Training $ix started! -- hn")

    # Reset model
    hn = HapticNet(; atype = a_type(Float32))

    # Train 3000 epochs in total, but take snapshot at every 1000 epochs
    for k in 1:3
        # Training routine
        res = train_epoch!(hn, dtrn, dtst; progress_bar = false, fig = false, info = true, epoch = 1000)
        # Save model
        save_as_jld2(hn, "hn-$ix-$k.jld2")
        # Add results to the ´results´vector
        push!(results, res)
    end
end


notify!("Training done! -- hn")
# Save results
JLD2.@save "results_hn.jld2" results = results
