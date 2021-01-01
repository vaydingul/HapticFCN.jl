push!(LOAD_PATH, "C://Users//volkan//Desktop//Graduate//Graduate_Era//Courses//COMP_541//Project//COMP541_Project//src//modules//")

using TUM69: NetworkData, kfold
using Network: VisualNet, train_generic!, GCN, train_epoch!

alexnet_PATH = "alexnet.mat"

nd = NetworkData("data/trial", "image"; read_type = "normal", batchsize = 2)

kf = kfold(nd; fold = 2)
epoch = 10

for (dtrn, dtst) in kf.folds[1:1]

    vn = VisualNet(alexnet_PATH)
    
    #for k = 1:epoch
    @show res = train_epoch!(vn, dtrn, dtst; progress_bar=true, fig=false, info=true, epoch=2)
    #end
    
end


