module Utils

using MultivariateStats: fit, PCA, transform


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
notify(str) = run(`curl https://notify.run/fnx04zT7QmOlLLa6 -d $str`)


end