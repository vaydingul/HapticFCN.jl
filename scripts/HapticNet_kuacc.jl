# Custom modules
using HapticFCN
#using HapticFCN: HapticNet, VisualNet, train_epoch!, save_as_jld2
#using HapticFCN: HapticData, VisualData
#using HapticFCN: notify!, a_type
# Ready-to-use modules
using CUDA

notify!("Script started! -- hn")


# Set path
path_train = CUDA.functional() ? "/userfiles/vaydingul20/data/new" : "./data/trial/accel/train" # path of the main data
path_test = CUDA.functional() ? "/userfiles/vaydingul20/data/new" : "./data/trial/accel/test" # path of the main data

notify!("HapticData is being constructed! -- hn")

hd_trn = HapticData(path_train; is_online = true, batchsize = 1)
hd_tst = HapticData(path_test; is_online = true, batchsize = 1)

hn = HapticNet(;atype = a_type(Float32))

res = train_epoch!(hn, hd_trn, hd_tst; progress_bar=true, fig=true, info=true, epoch=5)
 