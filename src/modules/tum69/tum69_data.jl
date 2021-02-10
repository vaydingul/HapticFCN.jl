export HapticData, VisualData




function HapticData(haptic_data_path::String; is_online = false, shuffle::Bool = false, batchsize = 200, partial::Bool = false,
    atype::Type = Array{Float32}, 
    freq_count=50, signal_count=300, Fs=10000, window_length=500, noverlap=400)

    haptic_dh = DataHandler(is_online, FunctionHolder(read_data, (), Dict()),
                                    FunctionHolder(readdlm, ('\n', Float32), Dict()))

    add_data_preprocess_method(haptic_dh, FunctionHolder(process_accel_signal, (), 
                                        Dict(:freq_count => freq_count, :signal_count => signal_count,
                                        :Fs => Fs, :window_length => window_length, :noverlap => noverlap)))


    haptic_nd = NetworkData(haptic_dh, haptic_data_path; shuffle = shuffle, batchsize = batchsize, partial = partial, atype = atype, xtype = atype )

end

function VisualData(visual_data_path::String; is_online = true, shuffle::Bool = false, batchsize = 200, partial::Bool = false,
    atype::Type = Array{Float32}, 
    crop_size = 384, resize_ratio = 0.5, o...)


    visual_dh = DataHandler(is_online, FunctionHolder(load_image_data, (), Dict(:type => type, :mode => mode)),
                                    FunctionHolder(load, (), Dict()))
    
    add_data_preprocess_method(visual_dh, FunctionHolder(process_image, (), Dict(:crop_size => crop_size, :resize_ratio => resize_ratio)))           
    add_data_preprocess_method(visual_dh, FunctionHolder(augment_image, (),  o...))

    visual_nd = NetworkData(visual_dh, visual_data_path; shuffle = shuffle, batchsize = batchsize, partial = partial, atype = atype, xtype = atype )

end
