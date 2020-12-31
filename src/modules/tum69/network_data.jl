export iterate, NetworkData, length

include("data_read_ops.jl")
include("..//preprocess//preprocess_ops.jl")


using Images
using DelimitedFiles
import Base: length, iterate, vcat

struct NetworkData

    ### Constructor inputs
    # main_path::String # Main path of the data
    # read_type::String # Basic or normal
    # train::Bool # Will train data be read?
    # test::Bool # Will test data be read?
    # concat_train_test::Bool # Is the concatenation of the train and test data required? (When k-fold is being applied)

    data::Array{Tuple{String,Int8}} # Paths of the individual data point
    type::String # Whether acceleration or image ==> "accel", "image"

    # Class of the individual data point, since it is Int value, it can be stored as variable directly

    # These two for the case of concatenation is not desired
    # data_train::Array{Tuple{String,Int8}}
    # data_test::Array{Tuple{String,Int8}}
    
    # Material dictionary
    material_dict::Dict{Int8,String}
    shuffle::Bool
    read_them_all::Bool # Whether all data will be read in once or will be iterated through
    # kfold::Bool # whether K-FOLD cross validation will be applied or not
    # kfold_size::Int # How many folds will be included in the cross validation ?
    batchsize::Int # Batchsize during training
    atype
end

function NetworkData(main_path, type; data_type="train", read_type::String="basic",shuffle::Bool=true, read_them_all::Bool=false,  batchsize::Int=10, atype=Array{Float32})
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
    
    NetworkData(data, type, material_dict, shuffle, read_them_all, batchsize, atype)


end


function NetworkData(data, nd::NetworkData)

    return NetworkData(data, nd.type, nd.material_dict, nd.shuffle, nd.read_them_all, nd.batchsize, nd.atype)

end

length(nd::NetworkData) = length(nd.data)

function iterate(nd::NetworkData, i=0)

    if length(nd) - i <= nd.batchsize
        return nothing
    end
    nexti = i + nd.batchsize

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
    
end