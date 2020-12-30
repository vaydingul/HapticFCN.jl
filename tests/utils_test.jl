push!(LOAD_PATH, "C://Users//volkan//Desktop//Graduate//Graduate_Era//Courses//COMP_541//Project//COMP541_Project//src//modules")
using Utils
using Random
using Test

notify!("Hi :)")
@test a_type(Float32) == Array{Float32}
pca_data = extract_PCA(randn(10, 10))
fs = kfold(randn(10, 10), randn(10))
