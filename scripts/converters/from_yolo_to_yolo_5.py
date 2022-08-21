import os


PATH_TO_YOLO_1_ANNOTATIONS = "./dataset/annotations"
PATH_TO_YOLO_2_ANNOTATIONS = "../../3-yolo-formated-datasets/5-license-plates-dataset/dataset/annotations"


def convert_from_yolo_to_yolo():
    if not os.path.exists(PATH_TO_YOLO_2_ANNOTATIONS):
        os.makedirs(PATH_TO_YOLO_2_ANNOTATIONS)
    else:
        print("A folder with annotations already exists.")
        answer = input("Do you want to overwrite it?[y/n]\n")
        if answer.casefold() != "y":
            exit()

    print("Start Processing...")

    for yolo_1_annotation_full_file_name in os.listdir(PATH_TO_YOLO_1_ANNOTATIONS):
        print(f"Processing: {yolo_1_annotation_full_file_name}")
        with open(f"{PATH_TO_YOLO_1_ANNOTATIONS}/{yolo_1_annotation_full_file_name}", "r") as yolo_1_annotation_file:
            yolo_2_annotation_full_file_name = yolo_1_annotation_full_file_name
            with open(f"{PATH_TO_YOLO_2_ANNOTATIONS}/{yolo_2_annotation_full_file_name}", "w") as yolo_2_annotation_file:
                for license_plate_info in yolo_1_annotation_file:
                    if license_plate_info.split(" ")[0] == "0":
                        print(license_plate_info, file=yolo_2_annotation_file, end="")

    print("Done Processing")

if __name__ == "__main__":
    convert_from_yolo_to_yolo()