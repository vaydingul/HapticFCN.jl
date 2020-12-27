push!(LOAD_PATH, "C://Users//volkan//Desktop//Graduate//Graduate_Era//Courses//COMP_541//Project//COMP541_Project//src//modules")
using Metrics
using Random

cm = confusion_matrix(rand(1:10, 10), rand(1:10, 10))
visualize(cm)