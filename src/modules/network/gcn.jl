export GCN

#using Knet: Data
#using Knet
#using Statistics

include("conv.jl")
include("genericmlp.jl")
include("network_ops.jl")


#include("..//TUM69.jl")
#using .TUM69: NetworkData



# Generic Convolutional Network (Convolution + Dense or Fully Convolutional)
struct GCN

    layers # Layers that will be included in GCN
    optimizer_type # Default optimizer that will be used during training
    lr # Default learning reate that will be used during training
    loss_fnc # Default loss function that will be used for training
    accuracy_fnc # Default accuracy function
    L1 # L1 normalization factor
    L2 # L2 normalization factor

end


function GCN(i_dim, o_dim, kernels; hidden=[], optimizer_type=adam, lr=1e-4, loss_fnc=nll4, accuracy_fnc=accuracy4, L1=0.0, L2=5e-4, atype=Array)
    # ´kernels´ field represents the specificaton of the Convolutional and Pooling layer properties
    # that will be passed to network
    #= 
        kernel[1] = Width of the filter
        kernel[2] = Height of the filter
        kernel[3] = Output dimension of the filter
        kernel[4] = Activation function that will be applied to the layer
        kernel[5] = Dropout probability that will applied to the layer
        kernel[6] = Padding of the input
        kernel[7] = Stride of the filter
        kernel[8] = Pooling window
        kernel[9] = Stride of the pooling window
        kernel[10] = Local response normalization option =#
    
    
    # Trivial dilation
    dilation = (1, 1)

    # Initialization of the layers vector
    layers = []

    x, y, C_x = i_dim # Spatial dimension and channel size of the input
    
    for kernel in kernels
        if length(kernel) == 10 # If length == 10, then it is custom network
            
            # Setting of the required parameters
            spatial_x = kernel[1] # Spatial dimension x of the filter
            spatial_y = kernel[2] # Spatial dimension y of the  filter
            C_y = kernel[3] # Output channel size of the filter
            f = kernel[4]
            p = kernel[5]
            padding_ = kernel[6]
            stride_ = kernel[7]
            pool_window_ = kernel[8]
            pool_stride_ = kernel[9]
            lrn_ = kernel[10]
            
            # Construct a new Convolution layer based on these parameters
            push!(layers, Conv(spatial_x, spatial_y, C_x, C_y;f=f, p=0, padding_=padding_,  stride_=stride_,
                pool_window_=pool_window_, pool_stride_=pool_stride_, lrn_=lrn_ ,atype=atype))
            
            # Dimension calculation of the output for each filter
            x = 1 + floor((x + 2 * padding_[1] - ((spatial_x - 1) * dilation[1] + 1)) / stride_[1])
            x = 1 + floor((x - pool_window_[1]) / pool_stride_[1])
            y = 1 + floor((y + 2 * padding_[2] - ((spatial_y - 1) * dilation[2] + 1)) / stride_[2])
            y = 1 + floor((y - pool_window_[2]) / pool_stride_[2])
            C_x = C_y # Input channel size of the new layer equals to output channel size of the previous layer

        elseif length(kernel) == 9 # If length == 9, then it is a pretrained network

            # Setting of the required parameters
            w_ = kernel[1]
            b_ = kernel[2]
            f = kernel[3]
            p = kernel[4]
            padding_ = kernel[5]
            stride_ = kernel[6]
            pool_window_ = kernel[7]
            pool_stride_ = kernel[8]
            lrn_ = kernel[9]

            spatial_x = size(w_, 1) # Spatial dimension x of the filter
            spatial_y = size(w_, 2) # Spatial dimension y of the  filter
            C_y = size(w_, 4) # Output channel size of the filter

            # Construct a new Convolution layer based on these parameters
            push!(layers, Conv(w_, b_;f=f, p=0, padding_=padding_,  stride_=stride_,
                pool_window_=pool_window_, pool_stride_=pool_stride_, lrn_=lrn_ ,atype=atype))
            
            # Dimension calculation of the output for each filter
            x = 1 + floor((x + 2 * padding_[1] - ((spatial_x - 1) * dilation[1] + 1)) / stride_[1])
            x = 1 + floor((x - pool_window_[1]) / pool_stride_[1])
            y = 1 + floor((y + 2 * padding_[2] - ((spatial_y - 1) * dilation[2] + 1)) / stride_[2])
            y = 1 + floor((y - pool_window_[2]) / pool_stride_[2])

            # C = 1 + floor(C + 2 * padding_[3] - ((C_x - 1) * dilation[1] + 1))
            # C_x = convert(Int64, 1 + floor(C_x + 2 * padding_[4] - ((C_y - 1) * dilation[1] + 1)))
            C_x = C_y # Input channel size of the new layer equals to output channel size of the previous layer

        end
    end
    
    i_dense = x * y * C_x # Inout dimension of the first Dense layer
    o_dense = o_dim # Output dimension of the MLP / end of the architecture
    
    if hidden == []
        # If hidden == [], then it is FCN!
        nothing
    else
        # Construction of MLP that will be added to the end of the Convolutional chain
        f_dense = kernels[end][4]
        p_dense = kernels[end][5]

        # Construct the MLP part of the CNN
        gmlp = GenericMLP(convert(Int64, i_dense), o_dense; hidden=hidden, f=f_dense, p=p_dense,
                        optimizer_type=optimizer_type, lr=lr, loss_fnc=loss_fnc, accuracy_fnc=accuracy_fnc, atype=atype)
        push!(layers, gmlp.layers...)
    end

    # Return the final CNN architecture
    GCN(Tuple(layers), optimizer_type, lr, loss_fnc, accuracy_fnc, L1, L2)
    
end

function (gcn::GCN)(x::Union{AbstractArray, KnetArray})
        # Feed-forward through MLP model (whole architecture)
    for l in gcn.layers
        
        x = l(x)
        
    end
        
    return x
        
end

    
function (gcn::GCN)(x, y)
        # Loss calculation for one batch
    loss = gcn.loss_fnc(gcn(x), y)
    if Knet.training() # Only apply regularization during training, only to weights, not biases.
        gcn.L1 != 0 && (loss += gcn.L1 * sum(sum(abs, l.w) for l in gcn.layers))
        gcn.L2 != 0 && (loss += gcn.L2 * sum(sum(abs2, l.w) for l in gcn.layers))
    end
    return loss 
end
    
function (gcn::GCN)(data::Data) 
        # Loss calculation for whole epoch/dataset
    return mean(gcn(x, y) for (x, y) in data)
        
end


function (gcn::GCN)(data::T) where T 
    # Loss calculation for whole epoch/dataset
return mean(gcn(x, y) for (x, y) in data)
end
