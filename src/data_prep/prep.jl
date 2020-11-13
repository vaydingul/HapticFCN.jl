#= 
    This script was written only-once execution. 
    It organizes the data, which is very messy.
    It basically walks around the all folders,
    collects data and put them in order via creating
    category-based folders

=#


#### Directories of new and old data folders
ACCEL_FOLDER_PATH_TRAIN = "./data/old/train/accel"
IMAGES_FOLDER_PATH_TRAIN = "./data/old/train/image"
NEW_PATH_ACCEL_TRAIN = "./data/new/train/accel"
NEW_PATH_IMAGES_TRAIN = "./data/new/train/image"
INSTANCE_COUNT_TRAIN = 10

ACCEL_FOLDER_PATH_TEST = "./data/old/test/accel"
IMAGES_FOLDER_PATH_TEST = "./data/old/test/image"
NEW_PATH_ACCEL_TEST = "./data/new/test/accel"
NEW_PATH_IMAGES_TEST = "./data/new/test/image"
INSTANCE_COUNT_TEST = 10


###### ACCELERATION DATA ORGANIZATION TRAIN ###################

files = readdir(ACCEL_FOLDER_PATH_TRAIN) # Fetch the files
material_dict = Dict{Int8, String}()
count = 0

for file in files

    material_name = join(split(file, "_")[begin:end-1]) # Learn material category from file name :(
    
    ## According to the filename, create a new folder and copy this file to there
    try 
        mkdir(joinpath(NEW_PATH_ACCEL_TRAIN, material_name))
        cp(joinpath(ACCEL_FOLDER_PATH_TRAIN, file), joinpath(NEW_PATH_ACCEL_TRAIN, material_name, file), force = true)
        global count += 1

        push!(material_dict, count => material_name)
    catch err
        cp(joinpath(ACCEL_FOLDER_PATH_TRAIN, file), joinpath(NEW_PATH_ACCEL_TRAIN, material_name, file), force = true)
    end
end


###### IMAGE DATA ORGANIZATION TRAIN ####################

images = readdir(IMAGES_FOLDER_PATH_TRAIN)

#print(images)
for k in 1:count
    
    start_ix = (k-1) * INSTANCE_COUNT_TRAIN + 1
    stop_ix = k * INSTANCE_COUNT_TRAIN

    ## In above loop, we created a Dict for material names. Now, we will use this Dict for images.
    for img_ix in start_ix:stop_ix

        try 
            mkdir(joinpath(NEW_PATH_IMAGES_TRAIN, material_dict[k]))
            cp(joinpath(IMAGES_FOLDER_PATH_TRAIN, images[img_ix]), joinpath(NEW_PATH_IMAGES_TRAIN, material_dict[k],  images[img_ix]), force = true)    
        catch err
            cp(joinpath(IMAGES_FOLDER_PATH_TRAIN, images[img_ix]), joinpath(NEW_PATH_IMAGES_TRAIN, material_dict[k],  images[img_ix]), force = true)
        end

    end

end



    
###### ACCELERATION DATA ORGANIZATION TEST ###################
files = readdir(ACCEL_FOLDER_PATH_TEST)
material_dict = Dict{Int8, String}()
count = 0
for file in files
    material_name = join(split(file, "_")[begin:end-1])
    try 
        mkdir(joinpath(NEW_PATH_ACCEL_TEST, material_name))
        cp(joinpath(ACCEL_FOLDER_PATH_TEST, file), joinpath(NEW_PATH_ACCEL_TEST, material_name, file), force = true)
        global count += 1

        push!(material_dict, count => material_name)
    catch err
        cp(joinpath(ACCEL_FOLDER_PATH_TEST, file), joinpath(NEW_PATH_ACCEL_TEST, material_name, file), force = true)
    end
end

###### IMAGE DATA ORGANIZATION TEST ####################

images = readdir(IMAGES_FOLDER_PATH_TEST)

#print(images)
for k in 1:count
    
    start_ix = (k-1) * INSTANCE_COUNT_TEST + 1
    stop_ix = k * INSTANCE_COUNT_TEST
    for img_ix in start_ix:stop_ix

        try 
            mkdir(joinpath(NEW_PATH_IMAGES_TEST, material_dict[k]))
            cp(joinpath(IMAGES_FOLDER_PATH_TEST, images[img_ix]), joinpath(NEW_PATH_IMAGES_TEST, material_dict[k],  images[img_ix]), force = true)    
        catch err
            cp(joinpath(IMAGES_FOLDER_PATH_TEST, images[img_ix]), joinpath(NEW_PATH_IMAGES_TEST, material_dict[k],  images[img_ix]), force = true)
        end

    end

end