module Network

import Knet # load, save
using Knet: conv4, pool, mat, KnetArray, nll, accuracy, zeroone, progress, progress!, sgd, adam, rmsprop,
             adagrad, param, param0, dropout, relu, minibatch, Data, sigm, tanh, save, load
using Statistics: mean
using Plots
using IterTools: ncycle, takenth
import .Iterators: cycle, Cycle, take
using Statistics: mean
using Base.Iterators: flatten
using StatsBase: mode

# Dense layer definition
struct Dense
    w # weight 
    b # bias 
    f # activation function
    p # dropout probability
end

# Constructor definition for Dense layer
Dense(i::Int,o::Int,f=relu; a_type = Array, pdrop=0) = Dense(param(o, i; atype = a_type), param0(o; atype = a_type), f, pdrop)
# Callable object that feed-forwards one minibatch through a layer
(d::Dense)(x) = d.f.(d.w * mat(dropout(x, d.p)) .+ d.b) 


# Struct that allows to create flexible MLP models
struct GenericMLP
    layers # List of layers that will be included in the MLP
    optimizer_type # Optimizer type that will be used in training
    lr # Learning rate that will be fed into optimizer
    function GenericMLP(i = 784, o = 10; hidden = [], f = relu, pdrop = 0.0, optimizer_type = sgd, lr = 0.1, a_type = Array)
        
        #=
        GenericMLP
            - It is a generic/flexible MLP constructor
            - It constructs the MLP according to the input, output, and hidden layer size.

        Example:
            GenericMLP(784, 10; hidden = [], f = identity, a_type = a_type, pdrop = 0, optimizer_type = sgd, lr = 0.15)

        Input:
            i = The size/length of the input data
            o = The size/length of the output data
            hidden = Layer size that will be placed between input and output layer
            f = Activation function: (relu, sigm, identity, tanh)
            pdrop = Dropout probability
            optimizer_type = The optimizer which will be used: (adam, sgd, rmsprop, adagrad)
            lr = Learning rate that will fed into optimizer
            a_type = Array type

        Output:
            layers = Constructed Dense layers
    =#
    
        architecture = vcat(i, hidden, o) # Architecture can be expressed as an 1D array
        layers = []

        for k in 1:size(architecture, 1) - 1
            # It recursively constructs the Dense layers
            push!(layers, Dense(architecture[k], architecture[k + 1], f, a_type = a_type, pdrop = pdrop)) 
        end

        new(Tuple(layers), optimizer_type, lr) 
        
    end
    
end



function (gmlp::GenericMLP)(x)
    # Feed-forward through MLP model (whole architecture)
    for l in gmlp.layers
    
        x = l(x)
    
    end
    
    return x
    
end

function (gmlp::GenericMLP)(x, y)
    # Loss calculation for one batch
    return nll(gmlp(x), y)
    
end

function (gmlp::GenericMLP)(data::Data)
    # Loss calculation for whole epoch/dataset
    return mean(gmlp(x, y) for (x, y) in data)
    
end


# Convolutional layer definition
struct Conv
    w # weight
    b # bias
    f # activation function
    p # dropout probability
    pool_opt # Whether pooling will take place or not
end

# Constructor definition for Convolutional layer
Conv(w1::Int,w2::Int,cx::Int,cy::Int,f=relu; a_type = Array, pdrop=0, pool_opt = true) = Conv(param(w1, w2, cx, cy; atype = a_type), param0(1, 1, cy, 1; atype = a_type), f, pdrop, pool_opt)
# Callable object that feed-forwards one minibatch
(c::Conv)(x) = c.pool_opt ? c.f.(pool(conv4(c.w, x, padding = (1,1)) .+ c.b)) : c.f.(conv4(c.w, dropout(x, c.p)) .+ c.b) 


struct GeneriCONV
    layers # List of layers that will be included in the CNN
    optimizer_type # Optimizer type that will be used in training
    lr # Learning rate that will be fed into optimizer
    function GeneriCONV(i_dim, o_dim, kernels; hidden = [], f = relu, pdrop = 0.0, optimizer_type = sgd, lr = 0.1, a_type = Array)
            
        #=
        GeneriCONV
            - It is a generic/flexible CNN constructor
            - It constructs the CNN according to the input, output,kernel and hidden layer size.

        Example:
            gconv4 = GeneriCONV(INPUT_DIM, 10, [(5, 20), (4, 50), (3, 100)]; 
                    hidden = [50],f = relu, a_type = a_type, pdrop = 0.0, 
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
            layers = Constructed Dense layers
    =#
    
        layers = []
        x, y, C_x = i_dim # Spatial dimension and channel size of the input
        
        for kernel in kernels
        
            spatial_x = kernel[1] # Spatial dimension of the square filter
            spatial_y = kernel[2] # Spatial dimension of the square filter
            C_y = kernel[3] # Output channel size of the square filter
            pool_opt = kernel[4]
            push!(layers, Conv(spatial_x, spatial_y, C_x, C_y, f; a_type = a_type, pdrop = pdrop, pool_opt = pool_opt))
            
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
            gmlp = GenericMLP(convert(Int64, i_dense), o_dense; hidden = hidden, f = f, a_type = a_type, pdrop = pdrop)
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
    
end

function nll4(x, y)

    #=
    This function execute following processes:
        - It calculates ´nll´ of 4D tensor output
      
    Usage:
        nll(x, y)

    Input:
        x = Output of the network, dense prediction as 4D tensor
        y = True label of the corresponding x
    

    Output:
        loss = Calculated loss value
    =#

    x = permutedims(x, (3,1,2,4))
    sc, sx, sy, sn = size(x)
    y_ = vcat(collect(fill(y[k], sx * sy ) for k in 1:sn)...)
    loss = nll(mat(x, dims = 1), y_)
    return loss

end


function max_vote(y)
    y = getindex.(argmax(y, dims = 1), 1)
    u = unique(y)
    d=Dict([(i,count(x->x==i,y)) for i in u])
    argmax(d)
    
    #mode(y)
end


function _accuracy4(x, y; average = true)

    #=
    This function execute following processes:
        - It calculates accuracy of the model for given x and y value
        - If average == true, then it gives directly the accuracy,
            if it is not, then it gives correct number of predictions and total count as 
            2-element Tuple.
      
    Usage:
        _accuracy4(x, y)

    Input:
        x = Output of the network, dense prediction as 4D tensor
        y = True label of the corresponding x
    

    Output:
        _accuracy = Calculated accuracy
        or
        (correct_pred, total_count) = Number of correct predictions and total count as 
            2-element Tuple.
    =#

    x = permutedims(x, (3,1,2,4))
    sc, sx, sy, sn = size(x)
    correct = [max_vote(mat(x[:,:,:,k], dims = 1)) .== y[k] for k in 1:sn]
    average ? (sum(correct) / length(correct)) : (sum(correct), length(correct))

end



function accuracy4(model; data::Data)

    #=
    This function execute following processes:
        - It calculates accuracy of the model per batch for given model and Data object
        - 
    Usage:
        accuracy4(model; data = test_set)

    Input:
        model = Network model to be evaluated
        data = Batch to be processed in model
    

    Output:
        accuracy = Calculated accuracy
    =#
    correct = 0.0
    count = 0.0

    for (x, y) in data

        (corr, cnt) = _accuracy4(model(x), y; average = false)
        correct += corr
        count += cnt
    end

    accuracy = correct/count
    return accuracy
end



function train_summarize!(model, dtrn, dtst; train_type = "epoch", fig = true, info = true, epoch = 100, conv_epoch = 50, max_conv_cycle = 20)
    #=
        train_summarize
            - It trains the given model
            - At the end of the training, it displays summary-like information of the training and the model

        Example:
            res_conv_4 = train_summarize(gconv4, dtrn, dtst; 
                            train_type = "converge", fig = true, info = true, 
                            epoch = 100, conv_epoch = 50, max_conv_cycle = 6);

        Input:
            model = The NN model that will be trained
            dtrn = Train data
            dtst = Test data
            train_type = It determines whether training will be based on a given epoch number or a condition.
                        If train_type = "epoch", then the model will be trained (epoch) times.
                        If train_type = "converge", the the model will be trained until
                            %100 accuracy is obtained.
            fig = Figure setting
                  If fig = true, then two figure will be displayed after training,
                  If fig = false, then no figure will be displayed.
            info = Text-based information setting
                  If info = true, then a text-based information will be displayed after training,
                  If info = false, then no text-based information will be displayed.
    
            epoch = If the train_type = "epoch", then it is number of epoch that the model will be trained.
    
            conv_epoch = If the train_type = "converge", then, in each trial, the model will be trained (conv_epoch) times,
                        and, then, the accuracy will be checked for termination of the iteration
    
            max_conv_cycle = If the train_type = "converge", and, still, the accuracy is not %100 when the model
                            has already been trained (max_conv_cycle * conv_epoch) times, then the iteration will be terminated.

        Output:
            result = Loss and misclassification errors of train and test dataset
    =#
    
    if train_type == "epoch"
        # Number of (epoch) times training
        result = ((model(dtrn), model(dtst), 1.0-accuracy4(model; data = dtrn), 1.0-accuracy4(model; data = dtst)) 
                for x in takenth(progress(model.optimizer_type(model,ncycle(dtrn,epoch), lr = model.lr)),length(dtrn)));

        result = reshape(collect(Float32,flatten(result)),(4,:));  
        
    elseif train_type == "converge"
        
        result = [];
        
        # Training until %100 accuracy
        while accuracy4(model; data = dtrn) != 1.0 && max_conv_cycle > 0
            
            res = ((model(dtrn), model(dtst), 1-accuracy4(model; data = dtrn), 1-accuracy4(model; data = dtst))
                for x in takenth(progress(model.optimizer_type(model,ncycle(dtrn,conv_epoch), lr = model.lr)),length(dtrn)));
            
            max_conv_cycle -= 1;
            
            push!(result, reshape(collect(Float32,flatten(res)),(4,:)));
            
        end
        
        result = hcat(result...)
        
    end
    
    if fig 
        # Plotting
        display(plot([result[1,:], result[2,:]], xlabel = "Epoch", 
                title = "Loss", label = ["Train Loss" "Test Loss"]));
        
        display(plot([result[3,:], result[4,:]], xlabel = "Epoch", 
                title = "Misclassification Error",label = ["Train Misclassification Error" "Test Misclassification Error"]));

    end
    
    
    if info 
        # Text based information
        # Nothing but the redundantly placed print commands :)
        
        param_sum = 0;
        println("TRAINING PARAMETERS")
        println("\n")
        println("Activation Function = ", model.layers[1].f)
        println("Optimizer Type = ", model.optimizer_type)
        println("Learning Rate = ", model.lr)
        println("====================================================");
        println("LAYERS:");
        println("\n")
        
        # Calculation of the total number of parameters in the model
        for l in model.layers
            println(typeof(l)," ==> W = ",size(l.w),"   b = ",size(l.b));
            w = prod(size(l.w));
            b = prod(size(l.b));
            param_sum += w+b;
        end
        
        println("====================================================");
        println("In this network configuration,\nthere are total $param_sum parameters.");
        println("====================================================");
        println("Final Loss")
        println("Train Loss = ", result[1,end])
        println("Test Loss = ", result[2,end])
        println("\n")
        println("Final Misclassification Error")
        println("Train Misclassification Error = ", result[3,end])
        println("Test Misclassification Error = ", result[4,end])
        println("\n")
        println("Test Accuracy = ", accuracy4(model; data = dtst))
        println("====================================================");

        
        
    end
        
    return result
end











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

end
=#
end