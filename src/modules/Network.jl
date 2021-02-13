module Network

using FCN
import FCN: train_epoch!
using Statistics
using JLD2
using MAT
using Knet

include("./network/hn.jl"); export HapticNet
include("./network/vn.jl"); export VisualNet
include("./network/model_ops.jl"); export train_epoch!, save_as_jld2
end









######################## DEPRECATED CODE #####################################
#= 

# Constructor definition for Convolutional layer
Conv(w1::Int,w2::Int,cx::Int,cy::Int,f=relu; a_type=Array, pdrop=0, pool_opt=true) = Conv(param(w1, w2, cx, cy; atype=a_type), param0(1, 1, cy, 1; atype=a_type), f, pdrop, pool_opt)
# Callable object that feed-forwards one minibatch
(c::Conv)(x) = c.pool_opt ? c.f.(pool(conv4(c.w, x, padding=(1, 1)) .+ c.b)) : c.f.(conv4(c.w, dropout(x, c.p)) .+ c.b) 


struct GeneriCONV
    layers # List of layers that will be included in the CNN
    optimizer_type # Optimizer type that will be used in training
    lr # Learning rate that will be fed into optimizer
    function GeneriCONV(i_dim, o_dim, kernels; hidden=[], f=relu, pdrop=0.0, optimizer_type=sgd, lr=0.1, a_type=Array)
            
        #= 
        GeneriCONV
            - It is a generic/flexible CNN constructor
            - It constructs the CNN according to the input, output,kernel and hidden layer size.

        Example:
            gconv4 = GeneriCONV(INPUT_DIM, 10, [(5, 20), (4, 50), (3, 100)]; 
                    hidden = [50], f = relu, a_type = a_type, pdrop = 0.0, 
                    optimizer_type = adam, lr = 0.001)

        Input:
            i_dim = The size/length of the input data that will be fed into neural network
            o_dim = The size/length of the output data that will be fed into neural network (number of classes)
            kernel = The spatial dimension and nnumber of square filters
            hidden = Size of the layers that will be placed between output of the Convolutional chain and main output layer
            f = Activation function: (relu, sigm, identity, tanh)
            pdrop = Dropout probability
            optimizer_type = The optimizer which will be used: (adam, sgd, rmsprop, adagrad)
            lr = Learning rate that will fed into optimizer
            a_type = Array type

        Output:
            layers = Constructed Dense layers =#
    
        layers = []
        x, y, C_x = i_dim # Spatial dimension and channel size of the input
        
        for kernel in kernels
        
            spatial_x = kernel[1] # Spatial dimension of the square filter
            spatial_y = kernel[2] # Spatial dimension of the square filter
            C_y = kernel[3] # Output channel size of the square filter
            pool_opt = kernel[4]
            push!(layers, Conv(spatial_x, spatial_y, C_x, C_y, f; a_type=a_type, pdrop=pdrop, pool_opt=pool_opt))
            
            # Dimension calculation of the output for each filter
            x = pool_opt ? floor((x - spatial_x + 1) / 2) : x - spatial_x + 1
            y = pool_opt ? floor((y - spatial_y + 1) / 2) : y - spatial_y + 1

            C_x = C_y # Input channel size of the new layer equals to output channel size of the previous layer
        
        end
        
        i_dense = x * y * C_x # Inout dimension of the first Dense layer
        o_dense = o_dim # Output dimension of the MLP / end of the architecture
        
        if o_dim == 0
            # If o_dim == 0, then it is FCN!
            nothing
        else
            # Construction of MLP that will be added to the end of the Convolutional chain
            gmlp = GenericMLP(convert(Int64, i_dense), o_dense; hidden=hidden, f=f, a_type=a_type, pdrop=pdrop)
            push!(layers, gmlp.layers...)
        end
        new(Tuple(layers), optimizer_type, lr)
        
    
    end
    
    
end


function (gconv::GeneriCONV)(x)
    # Feed-forward through MLP model (whole architecture)
    for l in gconv.layers
    
        x = l(x)
    
    end
    
    return x
    
end

function (gconv::GeneriCONV)(x, y)
    # Loss calculation for one batch
    return nll4(gconv(x), y)
    
end

function (gconv::GeneriCONV)(data::Data)
    # Loss calculation for whole epoch/dataset
    return mean(gconv(x, y) for (x, y) in data)
    
end =#




#= 
function LR_norm(x; atype = Array, o...)

    _, _, _, batch_size = size(x)
        
    if atype == Array
        for k in 1:batch_size
            
            x[:, :, :, k] = mapslices(x -> _LR_norm(x; o...), x[:, :, :, k], dims = 3)
            
        end
    else
        for k in 1:batch_size
            
            x[:, :, :, k] = atype(mapslices(x -> _LR_norm(x; o...), Array(x[:, :, :, k]), dims = 3))
            
        end
    end

    return x
end



function _LR_norm(x; k = 2, n = 5, alpha = 0.0001, beta = 0.75, atype = Array)
    
    nc = length(x)
    x_ = zeros(nc)
    #x_ = atype(zeros(nc))
    for i in 1:nc
        
        _lower = convert(Int, floor(max(1., i - n/2)))
        _upper = convert(Int, floor(min(nc, i + n/2)))
        _sum = sum(x[_lower:_upper].^2)
        x_[i] = x[i] ./ ((k .+ alpha .* _sum).^beta)
    end
    
    return x_
end 



function LR_norm(x; atype = Array{Float32}, o...)
    _, _, _, batch_size = size(x)
         
    x_ = []
            for k in 1:batch_size
                
               push!(x_, _LR_norm(x[: ,:, :, k]; atype = atype, o...))
   
            end

     x = cat(x_...; dims = 4)
        return x
end

function _LR_norm(x; atype =  Array{Float32}, o...)

    nx, ny, nc = size(x)
    
    x = mat(x; dims = 2)
    x_ = []#Array{atype}(undef, 0)
    for k = 1:(nx * ny)
        
    push!(x_, __LR_norm(x[k, :]; atype = atype, o...))
    
    end
x_ = cat(x_...; dims = 2)
x = reshape(x_', (nx, ny, nc))
return x
    
end

function __LR_norm(x; atype =  Array{Float32}, k = 2, n = 5, alpha = 0.0001, beta = 0.75)
    nc = size(x, 1)
    x_ = [] #atype(undef, 0)
    for i in 1:nc
        _lower = convert(Int, floor(max(1., i - n/2)))
        _upper = convert(Int, floor(min(nc, i + n/2)))
        _sum = sum(x[_lower:_upper].^2)
        push!(x_, x[i] ./ ((k .+ alpha .* _sum).^beta))
    
    end
    x = x_
    return atype(x)
end 


function LR_norm(x::T; o...) where T
    
    nx, ny, nc, batch_size = size(x)
    
    x = mat(permutedims(x, (3,1,2,4)); dims = 1)

    x = x'
    
    y = similar(x)

    x = getindex.([x], 1:size(x, 1), :)

    y = _LR_norm.(x; o...)

    y = vcat(y'...)

    y = reshape(y, (nx, ny, batch_size, nc))

    y = permutedims(y, (1,2,4,3))

    return y
end

function _LR_norm(x::T; k = 2, n = 5, alpha = 0.0001, beta = 0.75) where T

    k, n, alpha, beta = convert.(eltype(T), [k, n, alpha, beta]) 

    nc = length(x)

    _sum = []

        for i in 1:nc
            
            _lower = convert(Int, floor(max(1., i - n/2)))
            _upper = convert(Int, floor(min(nc, i + n/2)))
            push!(_sum,  sum(x[_lower:_upper].^2))
            
        end
    _sum = vcat(_sum...)
    return x./((k .+ alpha .* _sum).^beta)
end =#








#= 
function train!(model, train_data, test_data; period=10, iters=100)

    #=
    This function execute following processes:
        - It trains the model
        - It saves and returns the loss values

    Usage:
    train!(model, train_data, test_data; period=10, iters=100)

    Input:
    model = VisualNet or HapticNet model
    train_data = Collection of train minibatches
    test_data = Collection of test minibatches
    period = Sampling frequency of loss calculation
    iters = Total numbers of epoch

    Output:
    interval = Loss calculation interval
    train_loss = Loss values obtained by train minibatches
    test_loss = Loss values obtained by test minibatches
    =#

    train_loss = Array{Float64,1}() # Preallocation af array for future use
    test_loss = Array{Float64,1}() # Preallocation of array for future use
        
    for _ in 0:period:iters
    
        push!(train_loss, model(train_data)) # Calculate loss from feed-forwarding of train data
        push!(test_loss, model(test_data)) # Calculate loss from feed-forwarding of test data
        progress!(sgd(model, take(cycle(train_data), period))) # Train the network!
    
    end

    interval = 0:period:iters
    return interval, train_loss, test_loss
end



function accuracy(model, test_data)
    #=
    This function execute following processes:
        - It calculates the accuracy of the model via using test set

    Usage:
    accuracy(model, test_data)

    Input:
    model = VisualNet or HapticNet model
    test_data = Collection of test minibatches

    Output:
    accuracy = Accuracy of the network
    =#
    correct = 0.0
    count = 0.0
    for (x, y) in test_data

        y_pred = model(x) # Feed forward model on each minibatch

        max_ids = argmax(y_pred, dims=1) # Find the macimum valued indexes in One-Hot representation

        for (ix, max_id) in enumerate(max_ids)
            count += 1.0 # Count number of datapoint ==> TODO: Make it smarter... It is nonsense
            correct += max_id[1] == y[ix] ?  1.0 : 0.0  # Compare whether maximum valued index corresponds to 
                                                        # true label or not


        end
    end
    return correct / count

end =#

