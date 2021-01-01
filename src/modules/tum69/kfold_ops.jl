export kfold

using Random

struct kfold{T} 
    #= 

        K-FOLD cross validation seperator for the data =#

    folds::Array{Tuple{T, T}}

end



function kfold(X::D; fold=10) where D
    #= 
        General constructor for kfold_ struct

        kfold
            - It seperated the gicen data into kfold_ for training

        Example:
            kf = kfold(X_train, y_train; fold = 3, atype = a_type(Float32))

        Input:
            X = Input data of the model
            y = Desired output data of the model
            fold = Fold Construct
            minibatch_size = Minibatch size that will be included in each fold
            atype = Array type that will be passes
            shuffle = Shuffling option

        Output:
            result = Loss and misclassification errors of train and test dataset =#
    



    folds_ = Array{Tuple{D, D}}([])
    # Get size of the input data
    n = length(X.data)#[end]
    # We need to consider about sample size

    # Get permuted form of the indexes
    
    data_ = X.shuffle ? X.data[randperm(n)] : X.data

    # How many elements will be in one fold?
    # We are excluding the remaining elements
    fold_size = div(n, fold)


    for k in 1:fold

        # Lower and upper bounds of the folds
        l_test = (k - 1) * fold_size + 1
        u_test = k * fold_size

        tst = [l_test:u_test...]
        trn = [1:(l_test - 1)...,(u_test + 1):length(X.data)...]
        # Minibatching operation for each folding set
        push!(folds_, (D(data_[trn], X), D(data_[tst], X)))


    end

    # Return constructed kfold_ object
    kfold{D}(folds_)

end