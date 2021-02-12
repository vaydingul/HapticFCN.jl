# Custom modules
using HapticFCN.Network: HapticNet, VisualNet, train_epoch!, save_as_jld2
using HapticFCN.TUM69: HapticData, VisualData
using HapticFCN.Utils: notify!, a_type
# Ready-to-use modules
using CUDA
using Augmentor: FlipX, FlipY
notify!("Script started! -- vn")


# Set path
path = CUDA.functional() ? "/userfiles/vaydingul20/data/new" : "data/new/image/train" # path of the main data
DATA_PATH = isdir(path) && path
# Set pretrained network path
alexnet_PATH = CUDA.functional() ? "/scratch/users/vaydingul20/workfolder/COMP541_Project/alexnet.mat" : "alexnet.mat"  


# Augmentation pipeline
p1 = FlipX()
p2 = FlipY()
p3 = FlipX() |> FlipY()
notify!("VisualData is being constructed! -- vn")

vd = VisualData(path, p1, p2, p3; is_online = true, batchsize = 10)

vn = VisualNet(alexnet_PATH; atype=a_type(Float32))

res = train_epoch!(vn, vd, vd; progress_bar=true, fig=true, info=true, epoch=1)