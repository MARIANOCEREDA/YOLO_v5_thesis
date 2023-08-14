import os

def merge_txt_files(input_directory, output_filename):
    try:
        with open(output_filename, 'w') as output_file:
            for filename in os.listdir(input_directory):
                if filename.endswith(".txt"):
                    file_path = os.path.join(input_directory, filename)
                    with open(file_path, 'r') as input_file:
                        content = input_file.read()
                        output_file.write(content)
        print("Archivos combinados exitosamente.")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    input_directory = r"D:\facultad\Proyecto_FINAL\proyecto\YOLO_v5_thesis\dataset\package\labeled\labels"
    output_file = "summary.txt"  # Nombre del archivo de salida
    merge_txt_files(input_directory, output_file)
