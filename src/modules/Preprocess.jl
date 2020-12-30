module Preprocess

include("preprocess//preprocess_ops.jl"); export process_accel_signal, process_image, augment_image


end







#### DEPRECATED FUNCTIONS ###################3




#= 
function augment_image(X, y; o...)

    #=
    This function execute following processes:
        - It crops the given image to construct a tile view
        - It converts to Float32 representation in (W, H ,3) form
        - It rehapes to 4D form to be used in CNN
        

    Usage:
        process_image(X::Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2}, y; resize_ratio=0.1)

    Input:
        X = Input data
        y = Output data
        crop_size = Crop isze
    

    Output:
        X = Preprocessed X data
        y_new = Organized y data
    =#

    X = augment_image_X.(X; o...) # It applies the preprocessing to the all element

    y .+= 1 # Add 1 to output to be able to adapt to Knet

    # Since, the input data is splitted into parts, output data should be copied
    y_new = [fill(y[ix], size(x, 1)) for (ix,x) in enumerate(X)] 
    
    X = cat(X..., dims = 1) # Concatenate input data
    y_new = vcat(y_new...) # Concatenate output data
    return X, y_new

end

function augment_image_X(X; x_flip = true, y_flip = true, xy_flip = true)

    #=
    This function execute following processes:
        - It crops the given image to construct a tile view
        - It converts to Float32 representation in (W, H ,3) form
        - It rehapes to 4D form to be used in CNN
        

    Usage:
        process_image(X::Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2}, y; resize_ratio=0.1)

    Input:
        X = Input data
        y = Output data
        crop_size = Crop isze
    

    Output:
        X = Preprocessed X data
        y_new = Organized y data
    =#

     imgs = similar([], typeof(X), 4)


     imgs[1] = X
    
    if x_flip
         p = FlipX() 
         imgs[2] = augment(X, p)
    end
    
    if y_flip
         p = FlipY()
         imgs[3] = augment(X, p)
    end

    if xy_flip
         p = FlipX() |> FlipY()
         imgs[4] = augment(X, p)
    end

    return imgs


end =#




#= 
function augment_image_X2(X; x_flip = true, y_flip = true, xy_flip = true)

    #=
    This function execute following processes:
        - It crops the given image to construct a tile view
        - It converts to Float32 representation in (W, H ,3) form
        - It rehapes to 4D form to be used in CNN
        

    Usage:
        process_image(X::Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2}, y; resize_ratio=0.1)

    Input:
        X = Input data
        y = Output data
        crop_size = Crop isze
    

    Output:
        X = Preprocessed X data
        y_new = Organized y data
    =#
    
    imgs = similar([], typeof(X))


    push!(imgs, X)
    if x_flip
        push!(imgs,augment(X, FlipX()))
    end
    if y_flip
        push!(imgs,augment(X, FlipY()))
    end
    if xy_flip
        p = FlipX() |> FlipY()
        push!(imgs,augment(X, p))
    end

    return imgs


end =#



#= 
function process_image(X::Array{Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2}, 1}, y; crop_size = 384)

    #=
    This function execute following processes:
        - It crops the given image to construct a tile view
        - It converts to Float32 representation in (W, H ,3) form
        - It rehapes to 4D form to be used in CNN
        

    Usage:
        process_image(X::Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2}, y; resize_ratio=0.1)

    Input:
        X = Input data
        y = Output data
        crop_size = Crop isze
    

    Output:
        X = Preprocessed X data
        y_new = Organized y data
    =#

    X = process_image_X.(X; crop_size = crop_size) # It applies the preprocessing to the all element

    y .+= 1 # Add 1 to output to be able to adapt to Knet

    # Since, the input data is splitted into parts, output data should be copied
    y_new = [fill(y[ix], size(x, 4)) for (ix,x) in enumerate(X)] 
    
    X = cat(X..., dims = 4) # Concatenate input data
    y_new = vcat(y_new...) # Concatenate output data
    return X, y_new

end


function process_image_X(img::Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2}; crop_size = 384)

    #=
    This function execute following processes:
        - It crops the given image to construct a tile view
        - It converts to Float32 representation in (W, H ,3) form
        - It rehapes to 4D form to be used in CNN
        

    Usage:
        process_image_X(img::Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2}; resize_ratio=0.1)

    Input:
        img = Pure image data in the form of Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2}, which is Images.jl library output
        crop_size = Crop isze
    

    Output:
        img = Converted and resized image
    =#

    #img = imresize(img, ratio=resize_ratio) # Resize the image in the ratio of resize_ratio
     img = channelview(img) # Fetch its channels ==> R,G,B
     img = convert.(Float32, img) # Convert to Float32 representation ==> (W,H,3)
     img = permutedims(img, (2, 3, 1)) # Turn into the Knet applicable format
     img = split_into_patches(img, crop_size) # Split the image into patches to be able to increase dataset
    
    return img

end

function split_into_patches(img, crop_size)
    #=
    This function execute following processes:
        - It split image into patches to be able to increase the size of the image
        

    Usage:
        img = split_into_patches(img, crop_size) # Split the image into patches to be able to increase dataset
    
    Input:
        img = Image to be cropped
        crop_size = Size of each to be cropped image part

    Output:
        img_patches = The array of cropped images
    =#

    (W, H) = size(img)[1:2] # Get height and width of image

    # Find the integer divisor of height and width
    W_div = div(W, crop_size) 
    H_div = div(H, crop_size)

    img_patches = []

    for k in 1:W_div
        for m in 1:H_div
            
            # Minibatching-like operation to split image into square tiles
            x_lim = ((k-1) * crop_size + 1, k * crop_size);
            y_lim = ((m-1) * crop_size + 1, m * crop_size);
            push!(img_patches, reshape(img[x_lim[1]:x_lim[2], y_lim[1]:y_lim[2], 1:3], crop_size, crop_size, 3, 1))
       
        end
    end

    img_patches = cat(img_patches..., dims = 4) # Concatenate all of the square tiles
end =#




#=

function split_into_patches()


    #= 
    (W, H) = size(img)[1:2] # Get height and width of image

    # Find the integer divisor of height and width
    W_div = div(W, crop_size) 
    H_div = div(H, crop_size)

    etype = eltype(img)
    nd = ndims(img)
    img_patches = Array{Array{etype, nd + 1}}(undef, W_div * H_div)
    
    
    cnt = 1
    for k in 1:W_div
        for m in 1:H_div
            
            # Minibatching-like operation to split image into square tiles
            x_lim = ((k-1) * crop_size + 1, k * crop_size);
            y_lim = ((m-1) * crop_size + 1, m * crop_size);
            img_patches[cnt] = reshape(@view(img[x_lim[1]:x_lim[2], y_lim[1]:y_lim[2], :]), crop_size, crop_size, 3, 1)
            cnt += 1
        end
    end
    
    img_patches = cat(img_patches..., dims = 4) # Concatenate all of the square tiles =#

end
=#


#=

function process_image_X()

 img = imresize(img, ratio=0.5) # Resize the image in the ratio of resize_ratio
 img = channelview(img) # Fetch its channels ==> R,G,B
 img = convert.(Float32, img) # Convert to Float32 representation ==> (W,H,3)
 img = permutedims(img, (2, 3, 1)) # Turn into the Knet applicable format

end


=#