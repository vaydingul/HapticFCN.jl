push!(LOAD_PATH, "C://Users//volkan//Desktop//Graduate//Graduate_Era//Courses//COMP_541//Project//COMP541_Project//src//modules")
#using TUM69, Model, Utils
using TUM69: NetworkData, kfold
using Model: HapticNet, VisualNet, train_generic!
using Utils: notify!, a_type
alexnet_PATH = "alexnet.mat"

nd = NetworkData("data/new", "image"; read_type = "normal", batchsize = 2)

kf = kfold(nd; fold = 10)
epoch = 10

for (dtrn, dtst) in kf.folds

    vn = VisualNet(alexnet_PATH);
    println(typeof(dtrn))
    for k = 1:epoch

        tr_loss, ts_loss, tr_acc, ts_acc = train_generic!(vn, dtrn, dtst)

    end

end


