# Custom modules
using HapticFCN
#using HapticFCN: HapticNet, VisualNet, train_epoch!, save_as_jld2
#using HapticFCN: HapticData, VisualData
#using HapticFCN: notify!, a_type
# Ready-to-use modules
using CUDA
using Augmentor: FlipX, FlipY
notify!("Script started! -- vn")


# Set path
path_train = CUDA.functional() ? "/userfiles/vaydingul20/data/new" : "./data/trial/image/train" # path of the main data
path_test = CUDA.functional() ? "/userfiles/vaydingul20/data/new" : "./data/trial/image/test" # path of the main data

#DATA_PATH = isdir(path) && path
# Set pretrained network path
alexnet_PATH = CUDA.functional() ? "/scratch/users/vaydingul20/workfolder/COMP541_Project/alexnet.mat" : "./alexnet.mat"  


# Augmentation pipeline
p1 = FlipX()
p2 = FlipY()
p3 = FlipX() |> FlipY()
notify!("VisualData is being constructed! -- vn")

vd_trn = VisualData(path_train, p1; is_online = true, batchsize = 1)
vd_tst = VisualData(path_test, p1; is_online = true, batchsize = 1)
vn = VisualNet(alexnet_PATH; atype=a_type(Float32))

res = train_epoch!(vn, vd_trn, vd_tst; progress_bar=true, fig=true, info=true, epoch=1)
