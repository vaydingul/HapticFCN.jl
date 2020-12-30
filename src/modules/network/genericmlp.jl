export GenericMLP
include("dense.jl")
using Knet, Statistics
using Knet: Data


struct GenericMLP
    # Struct that allows to create flexible MLP models
    layers # List of layers that will be included in the MLP
    optimizer_type # Optimizer type that will be used in training
    lr # Learning rate that will be fed into optimizer
    loss_fnc # Default loss function that will be used during training
    accuracy_fnc # Accuracy function

    function GenericMLP(i=784, o=10; hidden=[], f=relu, p=0.0, optimizer_type=sgd, lr=0.1, loss_fnc=nll, accuracy_fnc=accuracy, atype=Array)
        
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
            push!(layers, Dense(architecture[k], architecture[k + 1]; f=f, p=p, atype=atype)) 
        end

        new(Tuple(layers), optimizer_type, lr, loss_fnc, accuracy_fnc) 
        
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
    return gmlp.loss_fnc(gmlp(x), y)
    
end

function (gmlp::GenericMLP)(data::Data)
    # Loss calculation for whole epoch/dataset
    return mean(gmlp(x, y) for (x, y) in data)
    
end
