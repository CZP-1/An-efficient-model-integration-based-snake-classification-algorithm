def freeze_backbone(model,model_name,freeze_layer):
    if model_name == "resnet50":
        for block in list(model.children())[:freeze_layer]: ## freeze_layer 5-8
            for param in list(block.parameters()):
                param.requires_grad = False

    elif model_name == "tf_efficientnet_b7_ns": ## freeze_layer 1-7
        for block in list(model.children())[:3]:
            for param in list(block.parameters()):
                param.requires_grad = False

        for block in list(model.children())[3][:freeze_layer]:
            for param in list(block.parameters()):
                param.requires_grad = False

    elif model_name == "tf_efficientnet_b5_ns": ## freeze_layer 1-7
        for block in list(model.children())[:3]:
            for param in list(block.parameters()):
                param.requires_grad = False

        for block in list(model.children())[3][:freeze_layer]:
            for param in list(block.parameters()):
                param.requires_grad = False

    elif model_name == "swin_large_patch4_window12_384":  ## freeze_layer 1-4
        for block in list(model.children())[:2]:
            for param in list(block.parameters()):
                param.requires_grad = False

        for block in list(model.children())[2][:freeze_layer]:
            for param in list(block.parameters()):
                param.requires_grad = False
    
    elif model_name == "swin_large_patch4_window7_224":  ## freeze_layer 1-4
        for block in list(model.children())[:2]:
            for param in list(block.parameters()):
                param.requires_grad = False

        for block in list(model.children())[2][:freeze_layer]:
            for param in list(block.parameters()):
                param.requires_grad = False

    elif model_name == "swin_base_patch4_window12_384":  ## freeze_layer 1-4
        for block in list(model.children())[:2]:
            for param in list(block.parameters()):
                param.requires_grad = False

        for block in list(model.children())[2][:freeze_layer]:
            for param in list(block.parameters()):
                param.requires_grad = False
                
    elif model_name == "convnext_large_384_in22ft1k":  ## freeze_layer 1-4
        for block in list(model.children())[:1]:
            for param in list(block.parameters()):
                param.requires_grad = False

        for block in list(model.children())[1][:freeze_layer]:
            for param in list(block.parameters()):
                param.requires_grad = False

    elif model_name == "resnest269e":
        for block in list(model.children())[:freeze_layer]: ## freeze_layer 5-8
            for param in list(block.parameters()):
                param.requires_grad = False

