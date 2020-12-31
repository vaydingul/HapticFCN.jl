export load_accel_data, load_image_data


using Images: load #, channelview, RGB, FixedPointNumbers, UInt8, Normed, ColorTypes
using DelimitedFiles: readdlm

function load_accel_data(filepath::String; type="train", mode::String="basic")

    #= 
    This function execute following processes:
        - Read TUM69 Haptic Database
        - Especially, time-series acceleration signals
        
    Usage:
    loaddata(data_directory; mode = "basic")

    Input:
    filepath = It must point to where the data is situated
    mode = It determines the reading mode.
        mode = "basic"  => Fetch 1 file from each folder [ Memory issues :( ]
        mode = "normal" => Fetch all data in each folder [ Full training ]

    Output:
    X_accel_train = Training input data of acceleration signals
    y_accel_train = Training output data of acceleration signals
    X_accel_test  = Testing input data of acceleration signals
    y_accel_test  = Testing output data of acceleration signals
    material_dict = Dictionary which maps material names to the integers =#

    ############# Preallocatzion of output arrays and dictionary ###############
    # X_accel_train = Array{Array{Float32,1},1}()
    # y_accel_train = Array{Int8,1}()
    # X_accel_test  = Array{Array{Float32,1},1}()
    # y_accel_test  = Array{Int8,1}()
    # material_dict = Dict{String,Int8}()

    train_data = Array{Tuple{String,Int8}}([])
    test_data = Array{Tuple{String,Int8}}([])
    material_dict = Dict{String,Int8}()

    ############################################################################


    ftypes = ["train", "test"] # Folder type specification 1
    dtype = "accel" # Folder type specification 2

    count = 1 # Since, it is 1, no need to add 1 to output data ´y´during preprocessing

    #= 
        Below DISGUSTING for loop basically walks around all folders and reads
        all necessary acceleration signal or camera imamge data. =#
    for ftype in ftypes

        f_path = joinpath(filepath, ftype)
        println(titlecase(ftype), " ", dtype, " data is being loaded!")
        
        d_path = joinpath(f_path, dtype)
        folders = readdir(d_path)

        for (ix, folder) in enumerate(folders)

            if haskey(material_dict, folder)

                nothing

            else

                push!(material_dict, folder => count)
                count += 1

            end

            cum_data_path = joinpath(d_path, folder)
            files = readdir(cum_data_path)

            for (ix2, file) in enumerate(files)
                    
                mode == "basic" && ix2 == 2 ? break : nothing 
                
                full_file_path = joinpath(cum_data_path, file)
             
                # data = readdlm(full_file_path, '\n', Float32)
                # data = reshape(data, size(data, 1))

                if ftype == "train"
                    
                    push!(train_data, (full_file_path, material_dict[folder]))
                    # push!(X_accel_train, data)
                    # push!(y_accel_train, material_dict[folder])
                        
                end

                if ftype == "test"

                    push!(test_data, (full_file_path, material_dict[folder]))
                    # push!(X_accel_test, data)
                    # push!(y_accel_test, material_dict[folder])
                        
                end

            end   
                
        end
                   
    end

    # Inverting the dictionary, to map materials based on integer values
    material_dict = Dict(value => key for (key, value) in material_dict)
    if type == "train"
        return train_data, material_dict

    elseif type == "test"
        return test_data, material_dict

    else
        return vcat(train_data, test_data), material_dict
    end
end

function load_image_data(filepath::String; type="train", mode::String="basic")

    #= 
    This function execute following processes:
        - Read TUM69 Haptic Database
        - Especially camera images
        
    Usage:
    loaddata(data_directory; mode = "basic")

    Input:
    filepath = It must point to where the data is situated
    mode = It determines the reading mode.
        mode = "basic"  => Fetch 1 file from each folder [ Memory issues :( ]
        mode = "normal" => Fetch all data in each folder [ Full training ]

    Output:

    X_image_train = Training input data of camera images
    y_image_train = Training output data of camera images
    X_image_test  = Testing input data of camera images
    y_image_test  = Testing output data of camera images
    material_dict = Dictionary which maps material names to the integers =#

    ############# Preallocatzion of output arrays and dictionary ###############

    # X_image_train = Array{Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2},1}() 
    # y_image_train = Array{Int8,1}()
    # X_image_test  = Array{Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2},1}()
    # y_image_test  = Array{Int8,1}()
    # material_dict = Dict{String,Int8}()

    train_data = Array{Tuple{String,Int8}}([])
    test_data = Array{Tuple{String,Int8}}([])
    material_dict = Dict{String,Int8}()
    ############################################################################


    ftypes = ["train", "test"] # Folder type specification 1
    dtype = "image" # Folder type specification 2

    count = 1

    #= 
        Below DISGUSTING for loop basically walks around all folders and reads
        all necessary acceleration signal or camera imamge data. =#
    for ftype in ftypes

        f_path = joinpath(filepath, ftype)
        println(titlecase(ftype), " ", dtype, " data is being loaded!")
        
        d_path = joinpath(f_path, dtype)
        folders = readdir(d_path)

        for (ix, folder) in enumerate(folders)

            if haskey(material_dict, folder)

                nothing

            else

                push!(material_dict, folder => count)
                count += 1

            end

            cum_data_path = joinpath(d_path, folder)
            files = readdir(cum_data_path)

            for (ix2, file) in enumerate(files)
                    
                mode == "basic" && ix2 == 2 ? break : nothing 
                
                full_file_path = joinpath(cum_data_path, file)
                
                # data = load(full_file_path)
                # data = channelview(data)
                # data = convert.(Float32, data)
                # data = reshape(data, (size(data, 2), size(data, 3), 3))
                    
                if ftype == "train"
                    
                    push!(train_data, (full_file_path, material_dict[folder]))
                    # push!(X_image_train, data)
                    # push!(y_image_train, material_dict[folder])
                        
                end

                if ftype == "test"

                    push!(test_data, (full_file_path, material_dict[folder]))
                    # push!(X_image_test, data)
                    # push!(y_image_test, material_dict[folder])

                end

            end
  
        end
            
    end

    # Inverting the dictionary, to map materials based on integer values
    material_dict = Dict(value => key for (key, value) in material_dict)
    if type == "train"
        return train_data, material_dict

    elseif type == "test"
        return test_data, material_dict

    else
        return vcat(train_data, test_data), material_dict
    end
end
    