include("../src/modules/TUM69.jl")
include("../src/modules/Preprocess.jl")
include("../src/modules/Network.jl")
include("../src/modules/Utils.jl")

## Third party packages
using PyPlot: plot, specgram, subplot, xlabel, ylabel, title, legend
using Images
using ImageView: imshow
using Knet: minibatch

## Handwritten modules
using .TUM69: loaddata   # Data reading
using .Preprocess: process_accel_signal, process_image # Preprocessing on the data
using .Network: Conv, Dense, VisualNet, train!, accuracy # Construction of custom network
using .Utils: plot_spectrogram 


data_path = "./../data/new" 

X_accel_train, y_accel_train,
X_accel_test, y_accel_test, 
X_image_train, y_image_train, 
X_image_test,y_image_test,
material_dict = loaddata(data_path; mode = "baseline" ); 


println("X_accel_train = ", summary(X_accel_train))
println("y_accel_train = ", summary(y_accel_train))
println("X_accel_test  = ", summary(X_accel_test))
println("y_accel_test  = ", summary(y_accel_test))
println("X_image_train = ", summary(X_image_train))
println("y_image_train = ", summary(y_image_train))
println("X_image_test  = ", summary(X_image_test))
println("y_image_test  = ", summary(y_image_test))
println("material_dict = ", summary(material_dict))

for k in 1:3
subplot(3,1,k)
plot(X_accel_train[k])
end

X_image_train[1]

X_image_train[2]

X_image_train[3]

plot_spectrogram(X_accel_train[1], 10000);

plot_spectrogram(X_accel_train[2], 10000);

plot_spectrogram(X_accel_train[3], 10000);

X_accel_train = process_accel_signal.(X_accel_train);
X_accel_test = process_accel_signal.(X_accel_test);

X_image_train = process_image.(X_image_train);
X_image_test = process_image.(X_image_test);

X_image_train = cat(X_image_train..., dims = 4); y_image_train .+= 1;
X_image_test = cat(X_image_test..., dims = 4); y_image_test .+= 1;
X_accel_train = cat(X_accel_train..., dims = 4);
X_accel_test = cat(X_accel_test..., dims = 4);

println("X_accel_train = ", summary(X_accel_train))
println("y_accel_train = ", summary(y_accel_train))
println("X_accel_test  = ", summary(X_accel_test))
println("y_accel_test  = ", summary(y_accel_test))
println("X_image_train = ", summary(X_image_train))
println("y_image_train = ", summary(y_image_train))
println("X_image_test  = ", summary(X_image_test))
println("y_image_test  = ", summary(y_image_test))
println("material_dict = ", summary(material_dict))

VNet =VisualNet(Conv(5,5,3,20), 
                Conv(5,5,20,50),
                Conv(5,5,50,100),
                Conv(5,5,100,200),
                Conv(5,5,200,400),
                Dense(7200,500,pdrop=0.3), 
                Dense(500,69,identity,pdrop=0.3))
summary.(l.w for l in VNet.layers)

dtrn = minibatch(X_image_train, y_image_train, 10)
dtst = minibatch(X_image_test, y_image_test, 10)


iters, train_loss, test_loss = train!(VNet, dtrn, dtst; period = 10, iters = 100)

plot(iters, train_loss, label = "Train Loss")
plot(iters, test_loss, label = "Test Loss")
xlabel("Epochs")
ylabel("Loss")
legend(loc = "best")
title("First Run Results")

accuracy(VNet, dtst)
