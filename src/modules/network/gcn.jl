export GCN

include("genericmlp.jl")
include("conv.jl")
include("network_ops.jl")
using Knet
using Statistics
using Knet: Data





struct GCN

    layers
    optimizer_type
    lr
    loss_fnc
    accuracy_fnc
    L1
    L2
    function GCN(i_dim, o_dim, kernels; hidden=[], optimizer_type=sgd, lr=0.1, loss_fnc=nll4, accuracy_fnc=accuracy4, L1=0.0, L2=5e-4, atype=Array)
        dilation = (1, 1)
        layers = []
        x, y, C_x = i_dim # Spatial dimension and channel size of the input
        # C = C_x
        
        for kernel in kernels
            if length(kernel) == 10

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

                push!(layers, Conv(spatial_x, spatial_y, C_x, C_y;f=f, p=0, padding_=padding_,  stride_=stride_,
                    pool_window_=pool_window_, pool_stride_=pool_stride_, lrn_=lrn_ ,atype=atype))
                
                # Dimension calculation of the output for each filter
                x = 1 + floor((x + 2 * padding_[1] - ((spatial_x - 1) * dilation[1] + 1)) / stride_[1])
                x = 1 + floor((x - pool_window_[1]) / pool_stride_[1])
                y = 1 + floor((y + 2 * padding_[2] - ((spatial_y - 1) * dilation[2] + 1)) / stride_[2])
                y = 1 + floor((y - pool_window_[2]) / pool_stride_[2])
                # C = 1 + floor(C + 2 * padding_[3] - ((C_x - 1) * dilation[1] + 1))
                # C_x = convert(Int64, 1 + floor(C_x + 2 * padding_[4] - ((C_y - 1) * dilation[1] + 1)))
                C_x = C_y # Input channel size of the new layer equals to output channel size of the previous layer

            elseif length(kernel) == 9

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
            gmlp = GenericMLP(convert(Int64, i_dense), o_dense; hidden=hidden, f=f_dense, p=p_dense,
                            optimizer_type=optimizer_type, lr=lr, loss_fnc=loss_fnc, accuracy_fnc=accuracy_fnc, atype=atype)
            push!(layers, gmlp.layers...)
        end
        new(Tuple(layers), optimizer_type, lr, loss_fnc, accuracy_fnc, L1, L2)
        
    end
end

function (gcn::GCN)(x)
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

