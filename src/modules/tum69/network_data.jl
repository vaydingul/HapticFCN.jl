export iterate, NetworkData, length

include("data_read_ops.jl")
include("preprocess_ops.jl")


using Images
using DelimitedFiles
import Base: length, iterate, vcat

abstract type GenericDataHandler end

struct TUM69DataHandler <: GenericDataHandler

    # Tuple of method and its arguments
    # The defined method should walk the directory of the data
    # and fetch the directories of the individual datum and label;
    # finally, should return an array of tuples consisting of the directories
    # and the labels.
    data_read_method 
    # Array of tuples consisting of functions and its arguments
    data_preprocess_method
    # Loading method
    is_online::Bool

end

# Helper functions which are specialized for this project!
function AccelerationSignalHandlerConstructor(; is_online = false, type = "train", mode = "basic", 
    freq_count=50, signal_count=300, Fs=10000, window_length=500, noverlap=400)

    data_read_method = [(load_accel_data, Dict(:type => type, :mode => mode))]

    data_preprocess_method = [(process_accel_signal, Dict(:freq_count => freq_count, :signal_count => signal_count,
    :Fs => Fs, :window_length => window_length, :noverlap => noverlap))]

    return TUM69DataHandler(data_read_method, data_preprocess_method, is_online)

end


function CameraImageHandlerConstructor(; is_online = true, type = "train", mode = "basic",
    crop_size = 384, resize_ratio = 0.5, o...)

    data_read_method = [(load_image_data, Dict(:type => type, :mode => mode))]

    data_preprocess_method = [(process_image, Dict(:crop_size => crop_size, :resize_ratio => resize_ratio)), 
    (augment_image, o...)]

    return TUM69DataHandler(data_read_method, data_preprocess_method, is_online)

end





mutable struct NetworkData{T} where T <: GenericDataHandler

    
    data::Array{Tuple{String,Int8}} # Paths of the individual data point and labels

    label_dict::Dict{Int8,String}
    shuffle::Bool
    read_rate
    read_count::Int # Whether all data will be read in once or will be iterated through
    
    batchsize::Int # Batchsize during training
    atype
    
    X_ # Temporary data fields to be stored in CPU
    y_ # Temporary data fields to be stored in CPU


end

function NetworkData{T}(main_path; shuffle::Bool=true, read_rate=1.0,  batchsize::Int=1, atype=Array{Float32}) where T<:GenericDataHandler
    #= 
         Custom constructor =#

    data , label_dict = T.data_read_method[1](main_path;T.data_read_method[2])

    read_count = floor(Int, length(data) * read_rate) # Number of data points to read each time
    
    # refresh_rate = floor(Int, read_count / batchsize)

    NetworkData{T}(data, label_dict, shuffle, read_rate,read_count, batchsize, atype, nothing, nothing)


end


function NetworkData{T}(data, nd::NetworkData{T}) where T <: GenericDataHandler

    read_count = floor(Int, length(data) * nd.read_rate) # Number of data points to read each time

    return NetworkData{T}(data, nd.label_dict, nd.shuffle, nd.read_rate,read_count, nd.batchsize, nd.atype, nd.nothing, nd.nothing)

end


function length(nd::NetworkData{T}) where T <: GenericDataHandler
     
    part = ceil(Int, length(nd.data) / nd.read_count)
    
    if nd.X_ !== nothing
        each_part = ceil(Int, length(nd.y_) / nd.batchsize)
        return part * each_part
    else
        return part
    end
    #=
    l = 0

    part_cnt, rem_cnt = divrem(length(nd.data), nd.read_count)
    
    l = ceil(Int, nd.read_count / nd.batchsize) * part_cnt
    
    l += ceil(Int, rem_cnt / nd.batchsize)

    return l
    =#


    #= 
    n = length(nd.data) / nd.batchsize
    ceil(Int,n) =#
end


function iterate(nd::NetworkData, state=(0, 0, true))

    s1, s2, s3 = state


    if nd.y_ !== nothing

        if (length(nd.data) - s1 <= 0) && (length(nd.y_) - s2 <= 0)

            return nothing
    
        end

        s2 = s2 % length(nd.y_)
        ps = length(nd.y_)

        # When we porocess an inout , it does not result in one-to-one relationship.
        # One inout may output as multiple modified version.
        # This state will check this situation.
        next_s2 = s2 + min(nd.batchsize, ps - s2)

        # This state is responsible for the data samples, which is one-to-one inherently.
        next_s1 = s3 ? min(s1 + nd.read_count, length(nd.data)) : s1 + 0
        #next_s1 = next_s2 == ps && s3 ? min(s1 + nd.read_count, length(nd.data)) : s1 + 0

        next_s3 = next_s2 == ps ? true : false

    else
        next_s2 = s2 + nd.batchsize

        # This state is responsible for the data samples, which is one-to-one inherently.
        next_s1 = s1 +  nd.read_count
    
        next_s3 = false

    end
    next_state = (next_s1, next_s2, next_s3)
    # nexti = i + min(nd.batchsize, length(nd.data) - i, nd.read_count - (i % nd.read_count))

    if s3

        y = vcat([nd.data[k][2] for k in s1 + 1:next_s1]...)
        println("Data reading...")
        if nd.type == "image"

            X = [load(nd.data[k][1]) for k in s1 + 1:next_s1]
            p1 = FlipX()
            p2 = FlipY()
            p3 = FlipX() |> FlipY()
            X, y = augment_image(X, y, p1, p2, p3)
            # Apply preprocessing on the images
            nd.X_, nd.y_ = process_image(X, y)
    
        else
    
            X = [vec(readdlm(nd.data[k][1], '\n', Float32)) for k in s1 + 1:next_s1]
            nd.X_, nd.y_ = process_accel_signal(X, y)

        end

    end

    # ids = [i + 1:nexti]

    Xbatch = convert(nd.atype, nd.X_[:,:,:,s2 + 1:next_s2])
    ybatch = nd.y_[s2 + 1:next_s2]

    #println.([state, next_state, length(nd.y_)])


    return ((Xbatch, ybatch), next_state)
    
end



#= 

function iterate(nd::NetworkData, i=0)

    if length(nd.data) - i <= 0

        return nothing

    end

    nexti = min(i + nd.batchsize, length(nd.data))
    #nexti = i + nd.batchsize

    y = vcat([nd.data[k][2] for k in i + 1:nexti]...)

    if nd.type == "image"

        X = [load(nd.data[k][1]) for k in i + 1:nexti]
        p1 = FlipX()
        p2 = FlipY()
        p3 = FlipX() |> FlipY()
        X, y = augment_image(X, y, p1, p2, p3)
        # Apply preprocessing on the images
        X, y = process_image(X, y)

    else

        X = [vec(readdlm(nd.data[k][1], '\n', Float32)) for k in i + 1:nexti]
        X, y = process_accel_signal(X, y)
    end

    #ids = [i + 1:nexti]

    X = convert(nd.atype, X)
    
    return ((X, y), nexti)
    
end =#