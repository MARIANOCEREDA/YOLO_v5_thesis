import os

def config():

    return {
        "device":"cuda",
        "num_anchors":9,
        "num_classes":1,
        "class_name":"package",
        "image_size":416,
        "train":{
            "batch_size":1,
            "lr":1e-5,
            "wd":1e-4,
            "epochs":50,
            "threshold":0.05,
            "iou_threshold":0.60,
            "nms_threshold":0.5
        },
        "paths":{
            "dataset":f"{os.getcwd()}\..\dataset\package\labeled",
            "train":{
                "images":f"{os.getcwd()}\..\dataset\package/train\images",
                "labels":f"{os.getcwd()}\..\dataset\package/train\labels",
                "csv_data":f"{os.getcwd()}\..\preprocessing\img_train_data.csv",
            },
            "validation":{
                "images":f"{os.getcwd()}\..\dataset\package/val\images",
                "labels":f"{os.getcwd()}\..\dataset\package/val\labels",
                "csv_data":f"{os.getcwd()}\..\preprocessing\img_val_data.csv",
            },
            "test":{
                "images":f"{os.getcwd()}\..\dataset\package/test\images",
                "labels":f"{os.getcwd()}\..\dataset\package/test\labels",
                "csv_data":f"{os.getcwd()}\..\preprocessing\img_test_data.csv",
            },
            "debug":{
                "images":f"{os.getcwd()}\..\dataset\package\debug\images",
                "labels":f"{os.getcwd()}\..\dataset\package\debug\labels",
                "csv_data":f"{os.getcwd()}\..\preprocessing\debug_img_data.csv",
            },
        },
        "anchors":[
            [(0.07477680953125,0.07449970689062499), (0.076011844828125,0.078125095234375), (0.097449782046875,0.094121923078125)],
            [(0.068528884609375,0.06776703846875), (0.06944086275,0.0684237843125), (0.071402693875,0.071131836734375)],
            [(0.055475632593750004,0.055650973000000006), (0.059375,0.057813), (0.061250111109375005,0.06235626667187501)]
           , 
        ],
        "detection_layers":[13, 26, 52]
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

    