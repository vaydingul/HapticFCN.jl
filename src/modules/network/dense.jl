export Dense


using Knet



# Dense layer definition
struct Dense
    w # weight 
    b # bias 
    f # activation function
    p # dropout probability
end


# Constructor definition for Dense layer
Dense(i::Int,o::Int; f=relu, p=0, atype=Array) = Dense(param(o, i; atype=atype), param0(o; atype=atype), f, p)
# Callable object that feed-forwards one minibatch through a layer
(d::Dense)(x) = d.f.(d.w * mat(dropout(x, d.p)) .+ d.b) 
