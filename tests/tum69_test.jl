push!(LOAD_PATH, "C://Users//volkan//Desktop//Graduate//Graduate_Era//Courses//COMP_541//Project//COMP541_Project//src//modules")
using TUM69

data_accel = load_accel_data("data/new"; mode = "baseline")
data_image = load_accel_data("data/new"; mode = "baseline")