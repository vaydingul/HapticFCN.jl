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
using Augmentor: FlipX, FlipY
notify!("Script started! -- vn")


# Set path
path = CUDA.functional() ? "/userfiles/vaydingul20/data/new" : "data/new" # path of the main data
DATA_PATH = isdir(path) && path
# Set pretrained network path
alexnet_PATH = "/scratch/users/vaydingul20/workfolder/COMP541_Project/alexnet.mat"

notify!("Data reading started! -- vn")

# Load image data
X_train, y_train, _, _, material_dict = load_image_data(DATA_PATH; mode="normal")

# Augmentation pipeline
p1 = FlipX()
p2 = FlipY()
p3 = FlipX() |> FlipY()
notify!("Augmentation started! -- vn")
X_train, y_train = augment_image(X_train, y_train, p1, p2, p3)


notify!("Preprocessing started! -- vn")
# Apply preprocessing on the images
# Here is where the code explodes
X_train, y_train = process_image(X_train, y_train)

# Seperate into 3 folds for training
kf = kfold(X_train, y_train, fold=3, atype=a_type(Float32))
results = []

notify!("Training started! -- vn")

for (ix, (dtrn, dtst)) in enumerate(kf.folds)
    
    notify!("Training $ix started! -- vn")

    #Rest the model
    vn = VisualNet(alexnet_PATH; atype=a_type(Float32))

    for k in 1:3
        # Training routine
        res = train_epoch!(vn, dtrn, dtst; progress_bar=false, fig=false, info=true, epoch=3000)
        # Save model 
        save_as_jld2(vn, "vn-$ix-$k.jld2")
        # Add results to the ´results´vector
        push!(results, res)

    end
    
end
notify!("Training done! -- vn")
# Save accumulated results
JLD2.@save "results_vn.jld2" results = results
