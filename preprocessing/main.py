from PIL import Image
from config.config import config as cfg
import csv
import os
import shutil

config_data = cfg()

labeled_images_path = config_data["paths"]["dataset"] + "/images"
labeled_labels_path = config_data["paths"]["dataset"] + "/labels"

all_images_files = os.listdir(labeled_images_path)

TRAIN_PERCENTAGE = 0.7
VAL_PERCENTAGE = 0.2
TEST_PERCENTAGE = 0.1

n_total_images = len(all_images_files) 
n_train_images = int(n_total_images * TRAIN_PERCENTAGE)
n_val_images = int(n_total_images * VAL_PERCENTAGE)
n_test_images = int(n_total_images * TEST_PERCENTAGE)

train_img_data_csv = "img_train_data.csv"
test_img_data_csv = "img_train_data.csv"
val_img_data_csv = "img_val_data.csv"

def get_destination_path(type:str, image_file:str, n_images:int):
    
    destination_img_path = ""
    destination_label_path = ""

    img_path = os.path.join(config_data["paths"][type]["images"], image_file)

    label_file_name = image_file.split(".")[0] + ".txt"

    label_path = os.path.join(config_data["paths"][type]["labels"], label_file_name)

    train_dir_len = len(os.listdir(config_data["paths"][type]["images"]))

    if os.path.exists(img_path) == False and train_dir_len < n_images:
        destination_img_path = img_path
    
    else:
        print(f"Image: {img_path} already exists in {type} path.")
    
    if os.path.exists(label_path) == False and train_dir_len < n_images:
        destination_label_path = label_path
    
    else:
        print(f"Label: {label_path} already exists in {type} path.")

    return destination_img_path, destination_label_path


def split_images_into_folders():

    print("Copying...")
    print(f"Train images: {n_train_images}")
    print(f"Validation images: {n_val_images}")
    print(f"Test images: {n_test_images}")

    for i, image_file in enumerate(all_images_files):

        source_folder = labeled_images_path
        source_path = os.path.join(source_folder, image_file)

        if i < n_train_images:

            destination_img_path, destination_label_path = get_destination_path("train", image_file, n_train_images)

        elif i < (n_train_images + n_val_images):

            destination_img_path, destination_label_path = get_destination_path("validation", image_file, n_val_images)

        elif i < (n_train_images + n_val_images + n_test_images):

            destination_img_path, destination_label_path = get_destination_path("test", image_file, n_test_images)

        if destination_img_path != "" and destination_label_path != "":

            try:
                shutil.copy(source_path, destination_img_path)
                print(f"Succesfully copied image to path: {destination_img_path}")

                shutil.copy(source_path, destination_label_path)
                print(f"Succesfully copied label to path: {destination_label_path}")

            except Exception as err:
                print(f"Error when trying to copy file : {err}")


def create_csv(images_folder:str, labels_folder:str, destination_file_name:str):

    img_data_matrix = []
    obj_data_matrix = []

    img_csv_headers = ["id", "width", "height", "source"]

    files = os.listdir(labels_folder)

    images = os.listdir(images_folder)

    for i in range(len(files)):

        # 1 - Data source
        img_src = images_folder + f"/{images[i]}"

        # 2 - get image id
        img_id = images[i].split('.')[0]

        # 3 - get image width and height
        img = Image.open(img_src)

        # 4 - Create image data list
        img_data_list = [img_id, img.height, img.width, img_src]

        img_data_matrix.append(img_data_list)

    with open(destination_file_name, 'w+', newline='') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(img_csv_headers)
        writer.writerows(img_data_matrix)


if __name__ == "__main__" : 

    split_images_into_folders()

    modes = ["train", "validation", "test"]

    for mode in modes:

        images_folder = config_data["paths"][mode]["images"]
        labels_folder = config_data["paths"][mode]["labels"]
        destination_csv_file = config_data["paths"][mode]["csv_data"]

        create_csv(images_folder, labels_folder, destination_csv_file)






    







