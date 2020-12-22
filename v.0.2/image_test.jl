using Images, CoordinateTransformations, Rotations, TestImages, OffsetArrays
using ImageView
using Augmentor
img = testimage("lighthouse")
imshow(img)
#=
img = channelview(img) # Fetch its channels ==> R,G,B
img = convert.(Float32, img) # Convert to Float32 representation ==> (W,H,3)
img = permutedims(img, (2, 3, 1)) # Turn into the Knet applicable format
=#
pl1 =  FlipY()
img1 = augment(img, pl1)


imshow.([img1])
