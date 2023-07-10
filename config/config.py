import os

def config():

    return {
        "num_anchors":9,
        "paths":{
            "dataset_path":f"{os.getcwd()}\..\dataset\labeled",
            "train_images_path":f"{os.getcwd()}\..\dataset\labeled\images",
            "train_labels_path":f"{os.getcwd()}\..\dataset\labeled\labels"
        }
    }

    