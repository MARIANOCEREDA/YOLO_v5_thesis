from PIL import Image
from config.config import config as cfg
import csv
import os


def create_csv(images_folder:str, labels_folder:str):

    img_data_csv = "img_data.csv"
    obj_data_csv = "objs_data.csv"

    img_data_matrix = []
    obj_data_matrix = []

    img_csv_headers = ["id", "width", "height", "bbox", "source"]
    obj_csv_headers = ["id", "class", "x_center", "y_center", "width", "height"]

    files = os.listdir(labels_folder)
    images = os.listdir(images_folder)

    for i in range(len(files)):

        # 1 - Data source
        img_src = images_folder + f"/{images[i]}"

        # 2 - get image id
        img_id = images[i].split('.')[0]

        # 3 - get image width and height
        img = Image.open(img_src)

        img_width = img.width

        img_height = img.height

        # 4 - Get bbox data
        with open(labels_folder + f"/{files[i]}", 'r') as f:
            
            file_content = f.readlines()

            for line in file_content:
                bbox_data = line.split(' ')
                obj_class, x_c, y_c, w, h = bbox_data

                # 5 - Data list
                img_data_list = [img_id, img_height, img_width, img_src]
                obj_data_list = [img_id, obj_class, x_c, y_c, w, h]

                # 6 - Append to Data matrix
                img_data_matrix.append(img_data_list)
                obj_data_matrix.append(obj_data_list)


    with open(obj_data_csv, 'w', newline='') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(img_csv_headers)
        writer.writerows(obj_data_matrix)

    with open(img_data_csv, 'w', newline='') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(img_csv_headers)
        writer.writerows(img_data_matrix)


if __name__ == "__main__" : 

    config_data = cfg()

    images_folder = config_data["paths"]["train_images_path"]
    labels_folder = config_data["paths"]["train_labels_path"]

    create_csv(images_folder, labels_folder)






    







