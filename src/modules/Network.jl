module Network

import Knet # load, save
using Knet: conv4, pool, mat, KnetArray, nll, zeroone, progress, sgd, param, param0, dropout, relu, minibatch, Data, progress!
using IterTools: ncycle, takenth
import .Iterators: cycle, Cycle, take
using Statistics: mean
using Base.Iterators: flatten

struct Conv; w; b; f; p; end
(c::Conv)(x) = c.f.(pool(conv4(c.w, dropout(x, c.p)) .+ c.b))
Conv(w1::Int,w2::Int,cx::Int,cy::Int,f=relu;pdrop=0) = Conv(param(w1, w2, cx, cy), param0(1, 1, cy, 1), f, pdrop)

struct Dense; w; b; f; p; end
(d::Dense)(x) = d.f.(d.w * mat(dropout(x, d.p)) .+ d.b) # mat reshapes 4-D tensor to 2-D matrix so we can use matmul
Dense(i::Int,o::Int,f=relu;pdrop=0) = Dense(param(o, i), param0(o), f, pdrop)

struct VisualNet
    layers
    VisualNet(layers...) = new(layers)
end
(vn::VisualNet)(x) = (for l in vn.layers; x = l(x); end; x)
(vn::VisualNet)(x,y) = nll(vn(x), y)
(vn::VisualNet)(d::Data) = mean(vn(x, y) for (x, y) in d)


function train!(model, train_data, test_data; period=10, iters=100)
    train_loss = Array{Float64,1}()
    test_loss = Array{Float64,1}()
        
    for _ in 0:period:iters
    
        push!(train_loss, model(train_data))
        push!(test_loss, model(test_data))
        progress!(sgd(model, take(cycle(train_data), period)))
    
    end
    
    return 0:period:iters, train_loss, test_loss
end

function accuracy(model, test_data)
    correct = 0.0
    count = 0.0
    for (x, y) in test_data

        y_pred = model(x)

        max_ids = argmax(y_pred, dims=1)

        for (ix, max_id) in enumerate(max_ids)
            count += 1.0
            correct += max_id[1] == y[ix] ?  1.0 : 0.0  


        end
    end
    return correct / count

end
end