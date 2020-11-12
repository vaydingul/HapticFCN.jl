include("../src/modules/TUM69.jl")
using .TUM69

data_path = "./../data/new"


X_accel_train, y_accel_train,
X_accel_test, y_accel_test, 
X_image_train, y_image_train, 
X_image_test,y_image_test,

material_dict = TUM69.loaddata(data_path)

print(summary.([X_accel_train, y_accel_train,
X_accel_test, y_accel_test, 
X_image_train, y_image_train, 
X_image_test,y_image_test,
material_dict]))

for (key,value) in material_dict
    println(key, " ", value)
end