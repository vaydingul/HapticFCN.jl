push!(LOAD_PATH, "C://Users//volkan//Desktop//Graduate//Graduate_Era//Courses//COMP_541//Project//COMP541_Project//src//modules")
using Model: HapticNet, VisualNet, train_epoch!, save_as_jld2
using TUM69: load_accel_data, load_image_data
using Preprocess: process_accel_signal, process_image, augment_image
using JLD2, Random
using Utils: kfold
using Test

hn = HapticNet()

JLD2.@save "trial.jld2" model = hn.model

hn2 = HapticNet("trial.jld2")

@test hn.model.layers[1].w == hn2.model.layers[1].w

vn = VisualNet("alexnet.mat")

JLD2.@save "trial2.jld2" model = vn.model

vn2 = VisualNet("trial2.jld2")

@test vn.model.layers[1].w == vn2.model.layers[1].w


xt,yt,_,_ = load_accel_data("data/trial"; mode = "baseline")
xt_, yt_, _, _ = load_image_data("data/trial"; mode = "baseline")
xt, yt = process_accel_signal(xt, yt)
xt_, yt_ = augment_image(xt_, yt_)
xt_, yt_ = process_image(xt_, yt_)

fs = kfold(xt, yt; fold = 2)
fs_ = kfold(xt_, yt_)

resvn = train_epoch!(vn, fs_.folds[1][1], fs_.folds[1][2]; fig = true, progress_bar = true, info = true, epoch = 1)
reshn = train_epoch!(hn, fs.folds[1][1], fs.folds[1][2]; fig = true, progress_bar = true, info = true, epoch = 1)

save_as_jld2(vn, "trial.jld2")
save_as_jld2(hn, "trial2.jld2")