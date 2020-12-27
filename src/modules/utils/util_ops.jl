export extract_PCA ,kfold, notify!, a_type


using MultivariateStats: fit, PCA, transform
using CUDA
using Knet: Data, minibatch, KnetArray
using Random
#using PyPlot: specgram, xlabel, ylabel
#using PyCall: pyimport
#=
function plot_spectrogram(data, fs)

    #=
    This function execute following processes:
        - Construct an spectrogram
        - Plot the spectrogram
        - It can be also output frequency domain data but, in the frame of PyPlot, it is not necessary,
            it is being done by DSP library.

    Usage:
    plot_spectrogram(data; fs)

    Input:
    data = Data array
    fs = Sampling frequency, which is needed for spectrogram coonstruction

    Output:
    []
    =#

    np = pyimport("numpy") # It must be imported to use Hamming window
    Pxx, freqs, bins, im = specgram(data, NFFT=500, noverlap=400, Fs=fs, window=np.hamming(500)); # Built-in spectrogram function
    xlabel("Time")
    ylabel("Frequency")

end 
=#

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

# Little notification tool :)
notify!(str) = run(`curl https://notify.run/fnx04zT7QmOlLLa6 -d $str`)
# Array type decider 
a_type(T) = (CUDA.functional() ? KnetArray{T} : Array{T})



struct kfold

    folds::Array{Tuple{Data, Data}}

end


function kfold(X, y; fold = 10, minibatch_size = 10, atype = Array{Float32}, shuffle = true)
    
    folds_ = Array{Tuple{Data, Data}}([])
    # Get size of the input data
    n = size(X)[end]
    # We need to consider about sample size



    # Reshape operation for generic dimensionalization
    X2 = reshape(X, :, n)
    y2 = reshape(y, :, n)

    # Get permuted form of the indexes
    perm_ixs = randperm(n)
    X2 = X2[:, perm_ixs]
    y2 = y2[:, perm_ixs]

    # How many elements will be in one fold?
    # We are excluding the remaining elements
    fold_size = div(n, fold)

    #X_ns = size(X)
    #X_ns[end] = fold_size

    #y_ns = size(y)
    #y_ns[end] = fold_size

    for k in 1:fold

        l_test = (k - 1) * fold_size + 1
        u_test = k * fold_size

        dtst = minibatch(X2[:,[l_test:u_test...]], y2[: ,[l_test:u_test...]], minibatch_size; xtype = atype, shuffle = shuffle, xsize = (size(X)[1:end-1]..., fold_size), ysize = (size(y)[1:end-1]..., fold_size))
        dtrn = minibatch(X2[:,[1:(l_test-1)...,(u_test+1):end...]], y2[: ,[1:(l_test-1)...,(u_test+1):end...]], minibatch_size; xtype = atype, shuffle = shuffle, xsize = (size(X)[1:end-1]..., fold_size), ysize = (size(y)[1:end-1]..., fold_size))
        push!(folds_, (dtrn, dtst))


    end


    kfold(folds_)

end







