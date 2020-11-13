module Utils


using PyPlot: specgram, xlabel, ylabel
using PyCall: pyimport

function plot_spectrogram(data, fs)

    np = pyimport("numpy")
    Pxx, freqs, bins, im = specgram(data, NFFT=500, noverlap=400, Fs=fs, window=np.hamming(500));
    xlabel("Time")
    ylabel("Frequency")

end 

end