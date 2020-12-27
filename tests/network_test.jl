push!(LOAD_PATH, "C://Users//volkan//Desktop//Graduate//Graduate_Era//Courses//COMP_541//Project//COMP541_Project//src//modules")
using Network: nll4, accuracy4, GCN, Conv, Dense, GenericMLP, train_epoch!
using Utils: notify!, kfold
using Random
using Knet: relu


loss = nll4(randn(10, 10, 10, 10), rand(1:10, 10))

cnv = Conv(10, 10, 10, 10);
dns = Dense(10 ,10)
gmlp = GenericMLP(10, 10)
gcn = GCN((10, 10, 10), 10, [(3, 3, 10, relu, 0.0, (1, 1), (1, 1), (2, 2),(2, 2), false)])

fs = kfold(randn(10,10,10,1000), rand(1:10, 1000))

acc = accuracy4(gcn; data = fs.folds[1][1])

res = train_epoch!(gcn, fs.folds[1][1], fs.folds[1][2]; fig = true, progress_bar = true, info = true, epoch = 5)