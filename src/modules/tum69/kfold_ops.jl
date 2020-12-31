
include("network_data.jl")

struct kfold_
    #= 

        K-FOLD cross validation seperator for the data =#

    folds::Array{Tuple{NetworkData, NetworkData}}

end



function kfold_(X::NetworkData; fold=10)
    #= 
        General constructor for kfold_ struct

        kfold_
            - It seperated the gicen data into kfold_ for training

        Example:
            kf = kfold_(X_train, y_train; fold = 3, atype = a_type(Float32))

        Input:
            X = Input data of the model
            y = Desired output data of the model
            fold = Fold Construct
            minibatch_size = Minibatch size that will be included in each fold
            atype = Array type that will be passes
            shuffle = Shuffling option

        Output:
            result = Loss and misclassification errors of train and test dataset =#
    



    folds_ = Array{Tuple{NetworkData, NetworkData}}([])
    # Get size of the input data
    n = length(X)[end]
    # We need to consider about sample size

    # Get permuted form of the indexes
    perm_ixs = randperm(n)
    X.data = X.data[perm_ixs]

    # How many elements will be in one fold?
    # We are excluding the remaining elements
    fold_size = div(n, fold)


    for k in 1:fold

        # Lower and upper bounds of the folds
        l_test = (k - 1) * fold_size + 1
        u_test = k * fold_size

        tst = [l_test:u_test...]
        trn = [1:(l_test - 1)...,(u_test + 1):length(nd)...]
        # Minibatching operation for each folding set
        push!(folds_, (NetworkData(X.data[trn], X), NetworkData(X.data[tst], X)))


    end

    # Return constructed kfold_ object
    kfold_{D}(folds_)

end