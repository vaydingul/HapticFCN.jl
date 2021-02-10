export notify!, a_type #kfold


using CUDA
using Knet



# Little notification tool :)
notify!(str) = run(`curl https://notify.run/fnx04zT7QmOlLLa6 -d $str`)
# Array type decider 
a_type(T) = (CUDA.functional() ? KnetArray{T} : Array{T})
