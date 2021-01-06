push!(LOAD_PATH, "C://Users//volkan//Desktop//Graduate//Graduate_Era//Courses//COMP_541//Project//COMP541_Project//src//modules//")

using TUM69: NetworkData, kfold
using Network: VisualNet, train_generic!, GCN, train_epoch!, HapticNet

alexnet_PATH = "alexnet.mat"

nd = NetworkData("data/trial", "accel"; read_type = "basic", batchsize = 2)

kf = kfold(nd; fold = 2)

for (dtrn, dtst) in kf.folds[1:1]

    hn = HapticNet()
    
    #for k = 1:epoch
    @show res = train_epoch!(hn, dtrn, dtst; progress_bar=true, fig=false, info=true, epoch=1)
    #end
    
end


