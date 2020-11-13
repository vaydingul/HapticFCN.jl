module Network

import Knet # load, save
using Knet: conv4, pool, mat, KnetArray, nll, zeroone, progress, sgd, param, param0, dropout, relu, minibatch, Data, progress!
using IterTools: ncycle, takenth
import .Iterators: cycle, Cycle, take
using Statistics: mean
using Base.Iterators: flatten

# Convolutional layer definition
struct Conv
    w 
    b
    f 
    p 
end

(c::Conv)(x) = c.f.(pool(conv4(c.w, dropout(x, c.p)) .+ c.b)) # Callable object that feed-forwards one minibatch
# Constructor definition for Convolutional layer
Conv(w1::Int,w2::Int,cx::Int,cy::Int,f=relu;pdrop=0) = Conv(param(w1, w2, cx, cy), param0(1, 1, cy, 1), f, pdrop)

# Dense layer definition
struct Dense
    w
    b
    f
    p
end

(d::Dense)(x) = d.f.(d.w * mat(dropout(x, d.p)) .+ d.b) # Callable object that feed-forwards one minibatch through a layer
# Constructor definition for Dense layer
Dense(i::Int,o::Int,f=relu;pdrop=0) = Dense(param(o, i), param0(o), f, pdrop)

# Custom VisualNet definition
struct VisualNet
    layers
    VisualNet(layers...) = new(layers)
end
(vn::VisualNet)(x) = (for l in vn.layers; x = l(x); end; x) # Callable object that feed-forwards one minibatch through a network
(vn::VisualNet)(x,y) = nll(vn(x), y) # Loss calculation for a minibatch
(vn::VisualNet)(d::Data) = mean(vn(x, y) for (x, y) in d) # Loss calculation for whole data


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
end