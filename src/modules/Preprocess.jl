module Preprocess

using Images: channelview, imresize, RGB, FixedPointNumbers, UInt8, Normed
using DSP: spectrogram, hamming, power, time, freq
using PyPlot
using PyCall

function process_accel_signal(acc_signal::Array{Float32,1}; freq_count=50, signal_count=1000, Fs=10000, window_length=500, noverlap=400)

       
    S = spectrogram(acc_signal, window_length, noverlap; fs=Fs, window=hamming)
    Pxx = power(S)
    Pxx = Pxx ./ maximum(Pxx)

    return reshape(Pxx[1:freq_count, 1:signal_count], freq_count, signal_count, 1, 1)
end

function process_image(img::Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2}; resize_ratio=0.1)

    img = imresize(img, ratio=resize_ratio)
    img = channelview(img)
    img = convert.(Float32, img)
    img = size(img, 2) > size(img, 3) ? reshape(img, (size(img, 2), size(img, 3), 3, 1)) : reshape(img, (size(img, 3), size(img, 2), 3, 1))

    return img

end


end