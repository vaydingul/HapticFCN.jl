module TUM69

using Images: channelview, imresize, RGB, FixedPointNumbers, UInt8, Normed
using Images

using DelimitedFiles

using GDH
using GDH: add_data_preprocess_method

using FunctionLib

using TiledIteration

using DSP: spectrogram, hamming, power, time, freq

using Augmentor

using MultivariateStats: fit, PCA, transform



include("./tum69/preprocess_ops.jl");
include("./tum69/tum69_data.jl"); export HapticData, VisualData

end