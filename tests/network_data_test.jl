push!(LOAD_PATH, "C://Users//volkan//Desktop//Graduate//Graduate_Era//Courses//COMP_541//Project//COMP541_Project//src//modules//")

using TUM69: NetworkData, kfold
using Network: VisualNet, train_generic!, GCN

alexnet_PATH = "alexnet.mat"

nd = NetworkData("data/trial", "image"; read_type = "normal", batchsize = 2)

kf = kfold(nd; fold = 2)
epoch = 1

for (dtrn, dtst) in kf.folds[1:1]

    vn = VisualNet(alexnet_PATH)
    
    for k = 1:epoch
    @show tr_loss, ts_loss, tr_acc, ts_acc = train_generic!(vn, dtrn, dtst)
    end
    
end


