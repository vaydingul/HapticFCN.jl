using Knet, Test, TiledIteration
using Random
crop_size = 384

img = randn(2800,3200,3)
(W, H, C) = size(img) # Get height and width of image

# Find the integer divisor of height and width
W_div = div(W, crop_size) 
H_div = div(H, crop_size)


@time itr = collect(TileIterator(axes(img[1:crop_size*W_div, 1:crop_size*H_div,:]), (crop_size,crop_size,C)))

@time a = cat(map(x -> reshape(img[x[1], x[2], :], (crop_size,crop_size,3,1)), itr)..., dims = 4);

#=

b = a 
b = permutedims(b, (1,3,2))
b = reshape(b, 12,4)

#b = reshape(b, 24,2)


@test b[:,:,1,1] == [1.0 1.0;
                    3.0 3.0];

@test b[:,:,2,1] == [9.0 9.0;
                    11.0 11.0];

@test b[:,:,3,1] == [17.0 17.0;
                    19.0 19.0];

                    =#