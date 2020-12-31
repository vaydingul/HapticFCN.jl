push!(LOAD_PATH, "C://Users//volkan//Desktop//Graduate//Graduate_Era//Courses//COMP_541//Project//COMP541_Project//src//modules")

using TUM69: NetworkData, kfold_

nd = NetworkData("data/trial", "accel"; batchsize = 2)

for (x,y) in nd
    summary.([x,y])
end