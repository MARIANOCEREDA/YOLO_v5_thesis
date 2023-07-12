import os

def config():

    return {
        "num_anchors":9,
        "num_classes":1,
        "image_size":416,
        "paths":{
            "dataset_path":f"{os.getcwd()}\..\dataset\labeled",
            "train_images_path":f"{os.getcwd()}\..\dataset\labeled\images",
            "train_labels_path":f"{os.getcwd()}\..\dataset\labeled\labels"
        }
    }

def architecture_config():
    """
    Tuples : (out channel, kernel_size, stride)
    List : [ "B", number_of_repeats] -> B is from "Block"
    U: Upsampling the feature map
    S: Scale prediction block
    B: Block
    """
    return [
        {"type":"Conv", "size":(32, 3, 1)},

        {"type":"Conv", "size":(64, 3, 2)},

        {"type":"Residual", "num_repeats":1},

        {"type":"Conv", "size":(128, 3, 2)},

        {"type":"Residual", "num_repeats":2},

        {"type":"Conv", "size":(256, 3, 2)},

        {"type":"Residual", "num_repeats":8},

        {"type":"Conv", "size":(512, 3, 2)},

        {"type":"Residual", "num_repeats":8},

        {"type":"Conv", "size":(1024, 3, 2)},

        {"type":"Residual", "num_repeats":4},  # To this point is Darknet-53

        {"type":"Conv", "size":(512, 1, 1)},

        {"type":"Conv", "size":(1024, 3, 1)},

        {"type":"Detect"},        # Detection layer / 32
        
        {"type":"Conv", "size":(256, 1, 1)},

        {"type":"Upsample"},

        {"type":"Conv", "size":(256, 1, 1)},

        {"type":"Conv", "size":(512, 3, 1)},

        {"type":"Detect"},        # Detection layer / 16

        {"type":"Conv", "size":(128, 1, 1)},

        {"type":"Upsample"},        

        {"type":"Conv", "size":(128, 1, 1)},

        {"type":"Conv", "size":(256, 3, 1)},

        {"type":"Detect"}]       # # Detection layer / 8

    