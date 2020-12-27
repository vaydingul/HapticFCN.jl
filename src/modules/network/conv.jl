export Conv

using Knet
include("network_ops.jl")


struct Conv
    w # weight
    b # bias
    f # activation function
    p # dropout probability
    padding_
    stride_
    pool_window_
    pool_stride_
    lrn_
    atype
end

# Constructor definition for Convolutional layer with given dimensions
function Conv(w1::Int,w2::Int,cx::Int,cy::Int; f=relu, p=0, padding_=(0, 0),  stride_=(1, 1),
            pool_window_=(2, 2), pool_stride_=(2, 2), lrn_=false ,atype=Array) 

    return Conv(param(w1, w2, cx, cy; atype=atype), param0(1, 1, cy, 1; atype=atype), f , p,
                padding_, stride_, pool_window_, pool_stride_, lrn_, atype)

end

# Constructor definition for Convolutional layer with given dimensions
function Conv(w, b; f=relu, p=0, padding_=(0, 0),  stride_=(1, 1),
    pool_window_=(2, 2), pool_stride_=(2, 2), lrn_=false ,atype=Array) 

    return Conv(param(w; atype=atype), param(reshape(b, (1, 1, size(b, 1), 1)); atype=atype), f , p,
        padding_, stride_, pool_window_, pool_stride_, lrn_, atype)

end


# Callable object that feed-forwards one minibatch
function (c::Conv)(x) 
    
    if c.lrn_

        x = c.f.(pool(conv4(c.w, dropout(x, c.p), padding=c.padding_, stride=c.stride_) .+ c.b; window=c.pool_window_, stride=c.pool_stride_))
        return LR_norm(x; atype=c.atype)
    
    else

        return c.f.(pool(conv4(c.w, dropout(x, c.p), padding=c.padding_, stride=c.stride_) .+ c.b; window=c.pool_window_, stride=c.pool_stride_))

    end


end