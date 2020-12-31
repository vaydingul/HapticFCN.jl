export LR_norm, nll4, accuracy4, _accuracy4, train_epoch!, train_generic!

using Knet, Plots
using Knet: Data

using IterTools: ncycle, takenth
using Base.Iterators: flatten


function LR_norm(x::T; k=2, n=5, alpha=0.0001, beta=0.75 , atype=Array{Float32}, el_type=Float32) where T

    #= 
    This function execute following processes:
        - It calculates ´Local Response Normalization´ of 4D tensor output
      
    Usage:
        LR_norm(x)

    Input:
        x = Input tensor
        k = Additive factor to the normalization
        n = Number of channels that will be taken into account
        alpha = Scale factor
        beta = Exponential factor
        atype = Working array type
        el_type = Working array element type
    

    Output:
        y = Calculated LRN value =#



    # Float32 conversion not to obtain Float64 at the end
    k, alpha, beta = convert.(el_type, [k, alpha, beta]) 

    nx, ny, nc, batch_size = size(x)

    # Take channels into front to apply convolution on them
    x = permutedims(x, (3, 1, 2, 4)) 
    x = reshape(x, (nc, nx * ny, 1, batch_size))

    kernel_size = convert(Int, n + 1)

    w = convert(atype, reshape(ones(el_type, kernel_size), (kernel_size, 1, 1, 1)))

    _sum = conv4(w, x.^2; padding=(convert(Int, ceil(n / 2)), 0))
    
    # If the sliding window is odd numbered, then 
    # we have one additional term at the end, which should be eliminated
    _sum = _sum[1:(end - mod(n, 2)), :, :, :]
    # LRN operation
    y = x ./ ((k .+ alpha .* _sum).^beta)

    # Resconstructiong the original shape
    y = reshape(y, (nc, nx, ny, batch_size))
    y = permutedims(y, (2, 3, 1, 4))

    return y
end

function nll4(x, y)

    #= 
    This function execute following processes:
        - It calculates ´nll´ of 4D tensor output
      
    Usage:
        nll4(x, y)

    Input:
        x = Output of the network, dense prediction as 4D tensor
        y = True label of the corresponding x
    

    Output:
        loss = Calculated loss value =#

    x = permutedims(x, (3, 1, 2, 4))
    sc, sx, sy, sn = size(x)
    y_ = vcat(collect(fill(y[k], sx * sy) for k in 1:sn)...)
    loss = nll(mat(x, dims=1), y_)
    return loss

end


function max_vote(y)
    y = getindex.(argmax(y, dims=1), 1)
    u = unique(y)
    d = Dict([(i, count(x -> x == i, y)) for i in u])
    argmax(d)
    
    # mode(y)
end



function _accuracy4(x, y; average=true)
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
            2-element Tuple. =#

    x = permutedims(x, (3, 1, 2, 4))
    sc, sx, sy, sn = size(x)
    correct = [max_vote(mat(x[:,:,:,k], dims=1)) .== y[k] for k in 1:sn]
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
        accuracy = Calculated accuracy =#
    correct = 0.0
    count = 0.0

    for (x, y) in data

        (corr, cnt) = _accuracy4(model(x), y; average=false)
        correct += corr
        count += cnt
        
    end

    accuracy = correct / count
    return accuracy
end



function train_epoch!(model, dtrn, dtst; progress_bar=true, fig=true, info=true, epoch=100)
    #= 
        train_summarize
            - It trains the given model
            - At the end of the training, it displays summary-like information of the training and the model

        Example:
            res_conv_4 = train_summarize(gconv4, dtrn, dtst; 
                            progress_bar = true, fig = true, info = true, 
                            epoch = 100);

        Input:
            model = The NN model that will be trained
            dtrn = Train data
            dtst = Test data
          
            
            progress_bar = Progress bar setting

            
            fig = Figure setting
                  If fig = true, then two figure will be displayed after training,
                  If fig = false, then no figure will be displayed.


            info = Text-based information setting
                  If info = true, then a text-based information will be displayed after training,
                  If info = false, then no text-based information will be displayed.
    
            epoch = If the train_type = "epoch", then it is number of epoch that the model will be trained.
    
        Output:
            result = Loss and misclassification errors of train and test dataset =#
    

    if progress_bar
        result = ((model(dtrn), model(dtst), 1.0 - model.accuracy_fnc(model; data=dtrn), 1.0 - model.accuracy_fnc(model; data=dtst)) 
                for x in takenth(progress(model.optimizer_type(model, ncycle(dtrn, epoch), lr=model.lr)), length(dtrn)));
    else
        result = ((model(dtrn), model(dtst), 1.0 - model.accuracy_fnc(model; data=dtrn), 1.0 - model.accuracy_fnc(model; data=dtst)) 
                for x in takenth(model.optimizer_type(model, ncycle(dtrn, epoch), lr=model.lr), length(dtrn)));
    end

    result = reshape(collect(Float32, flatten(result)), (4, :));
        

    
    if fig 
        # Plotting
        display(plot([result[1,:], result[2,:]]; xlabel="Epoch", title="Loss", label=["Train Loss" "Test Loss"]));
        
        display(plot([result[3,:], result[4,:]]; xlabel="Epoch",title="Misclassification Error",label=["Train Misclassification Error" "Test Misclassification Error"]));

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
            println(typeof(l), " ==> W = ", size(l.w), "   b = ", size(l.b));
            w = prod(size(l.w));
            b = prod(size(l.b));
            param_sum += w + b;
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
        println("Test Accuracy = ", model.accuracy_fnc(model; data=dtst))
        println("====================================================");

        
        
    end
        
    return result
end



function train_generic!(model, dtrn, dtst; optimizer_type = nothing, lr = nothing)



    opt_ = optimizer_type === nothing ? model.optimizer_type : optimizer_type
    lr_ = lr === nothing ? model.lr : lr

    opt_(model, dtrn, lr= lr_);


    return model(dtrn), model(dtst), model.accuracy_fnc(dtrn), model.accuracy_fnc(dtst)

end

