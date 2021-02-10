export FusionNet


struct FusionNet
    #=

        Ready to use wrapper for the FusionNet

    =#

    haptic_model::HapticNet # GCN model that it wraps
    visual_model::VisualNet
end

function FusionNet(haptic_path::String, visual_path::String)
    #=

        Constructor definition for pretrained networks

    =#
    haptic_fname = endswith(haptic_path, ".jld2") && haptic_path
	visual_fname = endswith(visual_path, ".jld2") && visual_path

    JLD2.@load haptic_fname haptic_model
	JLD2.@load visual_fname visual_model
    FusionNet(haptic_model, visual_model)

end




