push!(LOAD_PATH, "C://Users//volkan//Desktop//Graduate//Graduate_Era//Courses//COMP_541//Project//COMP541_Project//src//modules")
using TUM69: load_accel_data, load_image_data
using Preprocess: process_accel_signal, process_image, augment_image    
using Utils: notify!
using Augmentor: FlipX, FlipY
# xt, yt, _, _ = load_accel_data("data/trial"; mode = "baseline")
xt_, yt_, _, _ = load_image_data("data/trial"; mode="baseline")
notify!("data read")
# xt, yt = process_accel_signal(xt, yt)
# notify!("process accel")

# @time xt_1, yt_1 = augment_image(xt_, yt_);
# p1 = FlipX()
# p2 = FlipY()
# p3 = FlipX() |> FlipY()
# @time xt_2, yt_2 = augment_image(xt_, yt_, p1, p2, p3)
# notify!("augment")

#@time xt_1, yt_1 = process_image(xt_, yt_)
xt_2, yt_2 = process_image2(xt_, yt_)
notify!("process image")


#augment(xt_[1], SplitChannels() |> PermuteDims(2,3,1) |> ConvertEltype(Float32))