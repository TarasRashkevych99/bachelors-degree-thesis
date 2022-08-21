import os
from PIL import Image

PATH_TO_CSV_ANNOTATIONS = "./dataset/annotations"
PATH_TO_CSV_IMAGES = "./dataset/images"
PATH_TO_YOLO_ANNOTATIONS = "../../3-yolo-formated-datasets/3-endtoend(lpd)/dataset/annotations"

classes = [
    "license-plate",
]

def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_from_csv_to_yolo():
    if not os.path.exists(PATH_TO_YOLO_ANNOTATIONS):
        os.makedirs(PATH_TO_YOLO_ANNOTATIONS)
    else:
        print("A folder with annotations already exists.")
        answer = input("Do you want to overwrite it?[y/n]\n")
        if answer.casefold() != "y":
            exit()

    print("Start Processing...")

    for csv_annotation_full_file_name in os.listdir(PATH_TO_CSV_ANNOTATIONS):
        print(f"Processing: {csv_annotation_full_file_name}")
        with open(f"{PATH_TO_CSV_ANNOTATIONS}/{csv_annotation_full_file_name}", "r") as csv_annotation_file:
            yolo_annotation_full_file_name = csv_annotation_full_file_name
            with open(f"{PATH_TO_YOLO_ANNOTATIONS}/{yolo_annotation_full_file_name}", "w") as yolo_annotation_file:
                for license_plate_info in csv_annotation_file:
                    license_plate_info = license_plate_info.split(" ")
                    xmin = float(license_plate_info[4])
                    xmax = float(license_plate_info[6])
                    ymin = float(license_plate_info[5])
                    ymax = float(license_plate_info[7])
                    csv_image_full_file_name = f"{csv_annotation_full_file_name.split('.')[0]}.jpg"
                    img = Image.open(f"{PATH_TO_CSV_IMAGES}/{csv_image_full_file_name}")
                    w, h = img.size
                    bb = convert((w, h), (xmin, xmax, ymin, ymax))
                    print("0" + " " + " ".join([str(a) for a in bb]), file=yolo_annotation_file)

    print("Done Processing")

if __name__ == "__main__":
    convert_from_csv_to_yolo()