export process_accel_signal, process_accel_signal_X, process_image, process_image_X, split_into_patches, augment_image, augment_image_X

include("..//utils//util_ops.jl")


using Images: channelview, imresize, RGB, FixedPointNumbers, UInt8, Normed
using DSP: spectrogram, hamming, power, time, freq
using Augmentor

function process_accel_signal(X::Array{Array{Float32, 1}, 1}, y::Array{Int8,1}; freq_count=50, signal_count=300, Fs=10000, window_length=500, noverlap=400)
    #=
    This function execute following processes:
        - It applies the preprocessing for all samples in the sample array
        - According to the result of the preprocessing of X, it organizes y

    Usage:
        process_accel_signal(acc_signal::Array{Array{Float32,1}, 1}, y::Array{Int8,1}; freq_count=50, signal_count=1000, Fs=10000, window_length=500, noverlap=400)

    Input:
        X = Input data
        y = Output data
        freq_count = Number of frequency point that will be returned
        signal_count = Number of time point that will be returned
        Fs = Sampling rate of the data
        window_length = FFT window length
        noverlap = Number of data point that will be overlapped furing FFT

    Output:
        X = Preprocessed X data
        y_new = Organized y data
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
        Pxx_PCA = (Pxx_PCA .- minimum(Pxx_PCA, dims = 1)) ./ (maximum(Pxx_PCA, dims = 1) .- minimum(Pxx_PCA, dims = 1)) # Normalization of the data
        return Pxx_PCA
    else
        Pxx = reshape(Pxx[1:freq_count, 1:signal_count], freq_count, signal_count, 1, 1) # Reshape to 4D form to be used in CNN
        Pxx = (Pxx .- minimum(Pxx, dims = 1)) ./ (maximum(Pxx, dims = 1) .- minimum(Pxx, dims = 1)) # Normalization of the data
        return Pxx
    end
end


function augment_image(X, y; o...)

    #=
    This function execute following processes:
        - It crops the given image to construct a tile view
        - It converts to Float32 representation in (W, H ,3) form
        - It rehapes to 4D form to be used in CNN
        

    Usage:
        process_image(X::Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2}, y; resize_ratio=0.1)

    Input:
        X = Input data
        y = Output data
        crop_size = Crop isze
    

    Output:
        X = Preprocessed X data
        y_new = Organized y data
    =#

    X = augment_image_X.(X; o...) # It applies the preprocessing to the all element

    y .+= 1 # Add 1 to output to be able to adapt to Knet

    # Since, the input data is splitted into parts, output data should be copied
    y_new = [fill(y[ix], size(x, 1)) for (ix,x) in enumerate(X)] 
    
    X = cat(X..., dims = 1) # Concatenate input data
    y_new = vcat(y_new...) # Concatenate output data
    return X, y_new

end

function augment_image_X(X; x_flip = true, y_flip = true, xy_flip = true)

    #=
    This function execute following processes:
        - It crops the given image to construct a tile view
        - It converts to Float32 representation in (W, H ,3) form
        - It rehapes to 4D form to be used in CNN
        

    Usage:
        process_image(X::Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2}, y; resize_ratio=0.1)

    Input:
        X = Input data
        y = Output data
        crop_size = Crop isze
    

    Output:
        X = Preprocessed X data
        y_new = Organized y data
    =#
    T = typeof(X)
    imgs = similar([], T)


    push!(imgs, X)
    if x_flip
        push!(imgs,augment(X, FlipX()))
    end
    if y_flip
        push!(imgs,augment(X, FlipY()))
    end
    if xy_flip
        p = FlipX() |> FlipY()
        push!(imgs,augment(X, p))
    end

    return imgs


end



function process_image(X::Array{Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2}, 1}, y; crop_size = 384)

    #=
    This function execute following processes:
        - It crops the given image to construct a tile view
        - It converts to Float32 representation in (W, H ,3) form
        - It rehapes to 4D form to be used in CNN
        

    Usage:
        process_image(X::Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2}, y; resize_ratio=0.1)

    Input:
        X = Input data
        y = Output data
        crop_size = Crop isze
    

    Output:
        X = Preprocessed X data
        y_new = Organized y data
    =#

    X = process_image_X.(X; crop_size = crop_size) # It applies the preprocessing to the all element

    y .+= 1 # Add 1 to output to be able to adapt to Knet

    # Since, the input data is splitted into parts, output data should be copied
    y_new = [fill(y[ix], size(x, 4)) for (ix,x) in enumerate(X)] 
    
    X = cat(X..., dims = 4) # Concatenate input data
    y_new = vcat(y_new...) # Concatenate output data
    return X, y_new

end


function process_image_X(img::Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2}; crop_size = 384)

    #=
    This function execute following processes:
        - It crops the given image to construct a tile view
        - It converts to Float32 representation in (W, H ,3) form
        - It rehapes to 4D form to be used in CNN
        

    Usage:
        process_image_X(img::Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2}; resize_ratio=0.1)

    Input:
        img = Pure image data in the form of Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2}, which is Images.jl library output
        crop_size = Crop isze
    

    Output:
        img = Converted and resized image
    =#

    #img = imresize(img, ratio=resize_ratio) # Resize the image in the ratio of resize_ratio
    img = channelview(img) # Fetch its channels ==> R,G,B
    img = convert.(Float32, img) # Convert to Float32 representation ==> (W,H,3)
    img = permutedims(img, (2, 3, 1)) # Turn into the Knet applicable format
    img = split_into_patches(img, crop_size) # Split the image into patches to be able to increase dataset
    
    return img

end

function split_into_patches(img, crop_size)
    #=
    This function execute following processes:
        - It split image into patches to be able to increase the size of the image
        

    Usage:
        img = split_into_patches(img, crop_size) # Split the image into patches to be able to increase dataset
    
    Input:
        img = Image to be cropped
        crop_size = Size of each to be cropped image part

    Output:
        img_patches = The array of cropped images
    =#

    (W, H) = size(img)[1:2] # Get height and width of image

    # Find the integer divisor of height and width
    W_div = div(W, crop_size) 
    H_div = div(H, crop_size)

    img_patches = []

    for k in 1:W_div
        for m in 1:H_div
            
            # Minibatching-like operation to split image into square tiles
            x_lim = ((k-1) * crop_size + 1, k * crop_size);
            y_lim = ((m-1) * crop_size + 1, m * crop_size);
            push!(img_patches, reshape(img[x_lim[1]:x_lim[2], y_lim[1]:y_lim[2], 1:3], crop_size, crop_size, 3, 1))
       
        end
    end

    img_patches = cat(img_patches..., dims = 4) # Concatenate all of the square tiles
end