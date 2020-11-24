module Utils

using MultivariateStats: fit, PCA, transform
using PyPlot: specgram, xlabel, ylabel
using PyCall: pyimport

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

function extract_PCA(data; max_out_dim = 50)

    M = fit(PCA, data; maxoutdim = max_out_dim, pratio = 1.0 )
    
    return transform(M, data)

end


end