module Preprocess

include("Utils.jl")

using Images: channelview, imresize, RGB, FixedPointNumbers, UInt8, Normed
using DSP: spectrogram, hamming, power, time, freq
using PyPlot
using PyCall
using .Utils: extract_PCA

function process_accel_signal(X::Array{Array{Float32, 1}, 1}, y::Array{Int8,1}; freq_count=50, signal_count=300, Fs=10000, window_length=500, noverlap=400)
    #=
    This function execute following processes:
        - It applies the preprocessing for all samples in the sample array
        - According to the result of the preprocessing of X, it organizes y

    Usage:
    process_accel_signal(acc_signal::Array{Array{Float32,1}, 1}, y::Array{Int8,1}; freq_count=50, signal_count=1000, Fs=10000, window_length=500, noverlap=400)

    Input:
    X = Sample array
    y = Output array
    freq_count = Number of frequency point that will be returned
    signal_count = Number of time point that will be returned
    Fs = Sampling rate of the data
    window_length = FFT window length
    noverlap = Number of data point that will be overlapped furing FFT

    Output:
    X = Preprocessed X data
    y = Organized y data
    =#

    X = process_accel_signal_X.(X; freq_count = freq_count, signal_count = signal_count, Fs = Fs, window_length = window_length, noverlap = noverlap)
    y .+= 1
    y_new = [fill(y[ix], size(x, 4)) for (ix,x) in enumerate(X)]
    
    X = cat(X..., dims = 4)
    y_new = vcat(y_new...)
    return X, y_new
end


function process_accel_signal_X(acc_signal::Array{Float32,1}; freq_count=50, signal_count=300, Fs=10000, window_length=500, noverlap=400)
    #=
    This function execute following processes:
        - Calculates spectrogram of time-series acceleration data
        - It normalizes the spectrogram
        - It crops the certain portion from frequency domain
        - It crops the certain portion from time domain
        - Finally, it reshapes the resultant data and returns as output

    Usage:
    process_accel_signal_X(acc_signal::Array{Float32,1}; freq_count=50, signal_count=1000, Fs=10000, window_length=500, noverlap=400)

    Input:
    acc_signal = Time-series acceleration data
    freq_count = Number of frequency point that will be returned
    signal_count = Number of time point that will be returned
    Fs = Sampling rate of the data
    window_length = FFT window length
    noverlap = Number of data point that will be overlapped furing FFT

    Output:
    Pxx = Reshaped, cropped spectrogram data
    =#
    S = spectrogram(acc_signal, window_length, noverlap; fs=Fs, window=hamming) # Built-in DSP library function
    Pxx = power(S) # Power content of the data
    Pxx_PCA = extract_PCA(Pxx)
    if size(Pxx_PCA, 1) == freq_count
        row, col = size(Pxx)
        col_reshape = divrem(col, signal_count)[1]
        Pxx_PCA = reshape(Pxx_PCA[:, 1:(signal_count * col_reshape)], freq_count ,signal_count, 1, col_reshape)
        Pxx_PCA = (Pxx_PCA .- minimum(Pxx_PCA)) ./ (maximum(Pxx_PCA) - minimum(Pxx_PCA)) # Normalization of the data
        return Pxx_PCA
    else
        Pxx = reshape(Pxx[1:freq_count, 1:signal_count], freq_count, signal_count, 1, 1) # Reshape to 4D form to be used in CNN
        Pxx = (Pxx .- minimum(Pxx)) ./ (maximum(Pxx) - minimum(Pxx)) # Normalization of the data
        return Pxx
    end
end

function process_image(img::Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2}; resize_ratio=0.1)

    #=
    This function execute following processes:
        - It resizes the given image
        - It converts to Float32 representation in (W, H ,3) form
        - Ot rehapes to 4D form to be used in CNN
        

    Usage:
    process_image(img::Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2}; resize_ratio=0.1)

    Input:
    img = Pure image data in the form of Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2}, which is Images.jl library output
    resize_ratio = Resizing ratio
    

    Output:
    img = Converted and resized image
    =#

    img = imresize(img, ratio=resize_ratio) # Resize the image in the ratio of resize_ratio
    img = channelview(img) # Fetch its channels ==> R,G,B
    img = convert.(Float32, img) # Convert to Float32 representation ==> (W,H,3)
    img = size(img, 2) > size(img, 3) ? reshape(img, (size(img, 2), size(img, 3), 3, 1)) : reshape(img, (size(img, 3), size(img, 2), 3, 1))
        # Finally reshape to 4D to be used in CNN
    return img

end


end