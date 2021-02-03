
struct TypedFunction{T <: Function,I,O} 
    f::T
end

(f::TypedFunction)(args...) = f.f(args...)

function tupleize(args) # I don't know how to go from (Int,Int) to Tuple{Int,Int}
    t = Tuple{Any}
    t.parameters = Core.svec(args...)
    t
end

tupleize(args::DataType) = Tuple{args}
tupleize(::Type{Tuple{}}) = Tuple{}

function return_type(f,args)
    codeinfo = Base.code_typed(f,args)[1]
    codeinfo.second
end

function TypedFunction(f, i, o, strict=false)
    
    m  = match_signature(f,i,o,strict)
    isempty(m) && error("No method matching signature found.")
    m = m[1]
    
    i = typeof(i) <: NTuple ? i : (i,)
    args = tupleize(i)
    
    ret = return_type(f,args)
    ret = typeof(ret) <: Tuple ? tupleize(ret) : ret
    
    TypedFunction{typeof(f),args,ret}(f)
end

import Base: show, match
show(io::IO,f::TypedFunction{T,I,O}) where {T,I,O} = write(io, string(f.f,": $(I) → $(O)")) 

function match_signature(m::Method,f::Function,i,o,comparator,strict)
    
    args = m.sig
    i = typeof(i) <: NTuple ? i : (i,)
    
    !comparator(tupleize( (typeof(f),i...) ),args) && return false
    
    ret = return_type(f,tupleize(i))
    ret = ret <: NTuple ? ret : tupleize(ret)
    !comparator(tupleize(o),ret) && return false
    
    true
end

function match_signature(f::Function,i,o,strict) 
    m  = collect(methods(f))
    comparator = strict ? (==) : (<:)
    filter(m->match_signature(m,f,i,o,comparator,strict),m)
end

f(z,x::Int,y::Int) = x+z
f(x::Int,y::Float64) = x+y
f(x::Int) = x
f() = 2
f(x::T,y::T) where T = 1
f(x::Task) = (1,2)
f(x::Vector) = ()


tf1 = TypedFunction(f, (Int,Float64),(Float64))
tf2 = TypedFunction(f, (Int),(Int))
tf3 = TypedFunction(f, (),(Int))
tf4 = TypedFunction(f, (Float64,Float64),(Int))
tf5 = TypedFunction(f, (Task),(Int,Int))
tf6 = TypedFunction(f, (Vector),())

g(f::TypedFunction{T,I,O}) where {T,I<:Tuple{Number,Float64},O<:Number} = 1
g(f::TypedFunction{T,I,O}) where {T,I<:Tuple{Number},O<:Number} = 2
g(f::TypedFunction{T,I,O}) where {T,I<:Tuple{},O<:Int} = 3

@assert g(tf1) == 1
@assert g(tf2) == 2
@assert g(tf3) == 3

match_signature(f,(Float64,Int,Int),(Float64),false)

match_signature(abs,(Int),(Int),false)