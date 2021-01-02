export iterate, NetworkData, length

include("data_read_ops.jl")
include("preprocess_ops.jl")


using Images
using DelimitedFiles
import Base: length, iterate, vcat

mutable struct NetworkData

    ### Constructor inputs
    # main_path::String # Main path of the data
    # read_type::String # Basic or normal
    # train::Bool # Will train data be read?
    # test::Bool # Will test data be read?
    # concat_train_test::Bool # Is the concatenation of the train and test data required? (When k-fold is being applied)

    data::Array{Tuple{String,Int8}} # Paths of the individual data point

    X_ # Temporary data fields to be stored in CPU
    y_ # Temporary data fields to be stored in CPU

    type::String # Whether acceleration or image ==> "accel", "image"
    material_dict::Dict{Int8,String}
    shuffle::Bool
    read_rate
    read_count::Int # Whether all data will be read in once or will be iterated through
    batchsize::Int # Batchsize during training
    atype
   
    # Class of the individual data point, since it is Int value, it can be stored as variable directly

    # These two for the case of concatenation is not desired
    # data_train::Array{Tuple{String,Int8}}
    # data_test::Array{Tuple{String,Int8}}
    
    # Material dictionary

    # kfold::Bool # whether K-FOLD cross validation will be applied or not
    # kfold_size::Int # How many folds will be included in the cross validation ?

end

function NetworkData(main_path, type; data_type="train", read_type::String="basic",shuffle::Bool=true, read_rate=1.0,  batchsize::Int=10, atype=Array{Float32})
    #= 
         Custom constructor =#

    if type == "accel"

        # data_train and data_test
        # train_data, test_data, material_dict = load_accel_data(main_path; mode=read_type)
        data, material_dict = load_accel_data(main_path; type=data_type, mode=read_type)
    else

        # data_train and data_test
        data, material_dict = load_image_data(main_path; type=data_type, mode=read_type)

    end

    # data, train_data, test_data = concat_train_test ? (vcat(train_data, test_data), nothing, nothing) : (nothing, train_data, test_data)
    # kfold_size = kfold ? kfold_size : nothing


    read_count = floor(Int, length(data) * read_rate) # Number of data points to read each time
    
    # refresh_rate = floor(Int, read_count / batchsize)

    NetworkData(data, nothing, nothing, type, material_dict, shuffle, read_rate, read_count, batchsize, atype)


end


function NetworkData(data, nd::NetworkData)

    read_count = floor(Int, length(data) * nd.read_rate) # Number of data points to read each time

    return NetworkData(data, nd.X_, nd.y_, nd.type, nd.material_dict, nd.shuffle, nd.read_rate, read_count, nd.batchsize, nd.atype)

end

function length(nd::NetworkData) 
    l = 0

    part_cnt, rem_cnt = divrem(length(nd.data), nd.read_count)
    
    l = ceil(Int, nd.read_count / nd.batchsize) * part_cnt
    
    l += ceil(Int, rem_cnt / nd.batchsize)

    return l



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

    println.([state, next_state, length(nd.y_)])


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