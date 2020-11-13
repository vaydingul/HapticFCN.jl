module TUM69

using Images: load, channelview, RGB, FixedPointNumbers, UInt8, Normed
using DelimitedFiles: readdlm

function loaddata(filepath::String; mode::String = "baseline")

    X_accel_train = Array{Array{Float32, 1}, 1}()
    y_accel_train = Array{Int8,1}()
    X_image_train = Array{Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2}, 1}() 
    y_image_train = Array{Int8,1}()
    X_accel_test  = Array{Array{Float32, 1}, 1}()
    y_accel_test  = Array{Int8,1}()
    X_image_test  = Array{Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2}, 1}()
    y_image_test  = Array{Int8,1}()
    material_dict = Dict{String,Int8}()

    ftypes = ["train", "test"]
    dtypes = ["accel", "image"]

    count = 0
    for ftype in ftypes
        f_path = joinpath(filepath, ftype)
        println(ftype)
        for dtype in dtypes
            println(dtype)
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
                    
                    mode == "baseline" && ix2 == 2 ? break : nothing 



                    full_file_path = joinpath(cum_data_path, file)

                    if dtype == "image"
                        data = load(full_file_path)
                        #data = channelview(data)
                        #data = convert.(Float32, data)
                        #data = reshape(data, (size(data, 2), size(data, 3), 3))
                    elseif dtype == "accel"
                        data = readdlm(full_file_path, '\n', Float32)
                        data = reshape(data, size(data, 1))
                    end


                    if ftype == "train"
                        if dtype == "accel"
                            push!(X_accel_train, data)
                            push!(y_accel_train, material_dict[folder])
                        end
                        if dtype == "image"
                            push!(X_image_train, data)
                            push!(y_image_train, material_dict[folder])
                        end
                    end

                    if ftype == "test"

                        if dtype == "accel"
                            push!(X_accel_test, data)
                            push!(y_accel_test, material_dict[folder])
                        end
                        if dtype == "image"
                            push!(X_image_test, data)
                            push!(y_image_test, material_dict[folder])

                        end
                    end
                end

                
                
            end
            
        
        end

    end

    return X_accel_train,y_accel_train,X_accel_test,y_accel_test,X_image_train,y_image_train,X_image_test,y_image_test,material_dict
end



end