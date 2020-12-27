push!(LOAD_PATH, "C://Users//volkan//Desktop//Graduate//Graduate_Era//Courses//COMP_541//Project//COMP541_Project//src//modules")
using TUM69
using Preprocess
using Utils: notify!

xt, yt, _, _ = load_accel_data("data/trial"; mode = "baseline")
xt_, yt_, _, _ = load_image_data("data/trial"; mode = "baseline")
notify!("data read")
xt, yt = process_accel_signal(xt, yt)
notify!("process accel")

xt_, yt_ = augment_image(xt_, yt_)
notify!("augment")

xt_, yt_ = process_image(xt_, yt_)
notify!("process image")

