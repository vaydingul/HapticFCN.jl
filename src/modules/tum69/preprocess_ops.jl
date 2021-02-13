export process_accel_signal, process_accel_signal_X, process_image, process_image_X, split_into_patches, augment_image, augment_image_X



function process_accel_signal(X, y; freq_count=50, signal_count=300, Fs=10000, window_length=500, noverlap=400)
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
        y_new = Organized y data =#
    X = vec.(X)
    #X = process_accel_signal_X.(X; freq_count=freq_count, signal_count=signal_count, Fs=Fs, window_length=window_length, noverlap=noverlap)
    X = map(x -> process_accel_signal_X(x; freq_count=freq_count, signal_count=signal_count, Fs=Fs, window_length=window_length, noverlap=noverlap), X)
    #y .+= 1
    y_new = [fill(y[ix], size(x, 4)) for (ix, x) in enumerate(X)]
    
    X = cat(X..., dims=4)
    y_new = vcat(y_new...)
    return X, y_new
end


function process_accel_signal_X(acc_signal; freq_count=50, signal_count=300, Fs=10000, window_length=500, noverlap=400)
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
        Pxx = Reshaped, cropped spectrogram data =#

    S = spectrogram(acc_signal, window_length, noverlap; fs=Fs, window=hamming) # Built-in DSP library function
    Pxx = power(S) # Power content of the data
    Pxx_PCA = extract_PCA(Pxx)
    if size(Pxx_PCA, 1) == freq_count
        row, col = size(Pxx)
        col_reshape = divrem(col, signal_count)[1]
        Pxx_PCA = reshape(Pxx_PCA[:, 1:(signal_count * col_reshape)], freq_count, signal_count, 1, col_reshape)
        Pxx_PCA = (Pxx_PCA .- minimum(Pxx_PCA, dims=1)) ./ (maximum(Pxx_PCA, dims=1) .- minimum(Pxx_PCA, dims=1)) # Normalization of the data
        return Pxx_PCA
    else
        Pxx = reshape(Pxx[1:freq_count, 1:signal_count], freq_count, signal_count, 1, 1) # Reshape to 4D form to be used in CNN
        Pxx = (Pxx .- minimum(Pxx, dims=1)) ./ (maximum(Pxx, dims=1) .- minimum(Pxx, dims=1)) # Normalization of the data
        return Pxx
    end
end

function extract_PCA(data; max_out_dim = 50)
    #=
    This function execute following processes:
        - It performs PCA on the given data

    Usage:
        plot_spectrogram(data; fs)

    Input:
        data = Data to be performed PCA
        max_out_dim = Maximum number of property to be extracted from PCA

    Output:
        data_transformed = Projected version of the data
    =#

    M = fit(PCA, data; maxoutdim = max_out_dim, pratio = 1.0 ) # PCA application
    data_transformed = transform(M, data) # Data projection

    return data_transformed

end



function augment_image(X, y, o...)

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
        y_new = Organized y data =#
    n_ops = length(o)
    x_dim = length(X)

    X = convert(Array{Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2}, 1}, X)

    X_ = X
    X_new = similar(X, (n_ops + 1) * x_dim)
    X_new[1:x_dim] = X
    for (ix, ops) in enumerate(o)
        # X_temp = similar(X)
        augmentbatch!(X, X_, ops)
        X_new[(ix) * x_dim + 1:x_dim * (ix + 1)] = X
        # X_new[(k-1) * 3 + 1: 3 * k] = X_temp
    end
        
    # push!(X_new, X_temp...) # It applies the preprocessing to the all element
    
    #y .+= 1 # Add 1 to output to be able to adapt to Knet
    y_new = vcat([y for _ in 1:(n_ops + 1)]...)
    
    return X_new, y_new

end


function process_image(X, y; crop_size=384, resize_ratio = 0.5)

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
        y_new = Organized y data =#



    X = map(x->process_image_X(x; crop_size = crop_size, resize_ratio = resize_ratio), X)
    #y .+= 1 # Add 1 to output to be able to adapt to Knet

    # Since, the input data is splitted into parts, output data should be copied
    y_new = [fill(y[ix], size(x, 4)) for (ix, x) in enumerate(X)] 
    
    X = cat(X..., dims=4) # Concatenate input data
    y_new = vcat(y_new...) # Concatenate output data
    return X, y_new

end


function process_image_X(img; crop_size=384, resize_ratio = 0.5)

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
        img = Converted and resized image =#

    img = augment(img, Scale(resize_ratio) |> SplitChannels() |> PermuteDims(2,3,1) |> ConvertEltype(Float32))
    img = split_into_patches(img, crop_size) # Split the image into patches to be able to increase dataset
    
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
        img_patches = The array of cropped images =#


    (W, H, C) = size(img) # Get height and width of image

# Find the integer divisor of height and width

    W_div = div(W, crop_size) 
    H_div = div(H, crop_size)
    itr = collect(TileIterator(axes(img[1:crop_size * W_div, 1:crop_size * H_div,:]), (crop_size, crop_size, C)))
    cat(map(x -> reshape(img[x[1], x[2], :], (crop_size, crop_size, 3, 1)), itr)..., dims=4)
end
