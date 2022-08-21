import xml.etree.ElementTree as ET
import os


PATH_TO_VOC_ANNOTATIONS = "./dataset/annotations"
PATH_TO_YOLO_ANNOTATIONS = "../../3-yolo-formated-datasets/1-car-plate-detection/dataset/annotations"

classes = [
    "licence",
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


def convert_from_voc_to_yolo():
    if not os.path.exists(PATH_TO_YOLO_ANNOTATIONS):
        os.makedirs(PATH_TO_YOLO_ANNOTATIONS)
    else:
        print("A folder with annotations already exists.")
        answer = input("Do you want to overwrite it?[y/n]\n")
        if answer.casefold() != "y":
            exit()

    print("Start Processing...")

    for voc_annotation_full_file_name in os.listdir(PATH_TO_VOC_ANNOTATIONS):
        print(f"Processing: {voc_annotation_full_file_name}")
        with open(f"{PATH_TO_VOC_ANNOTATIONS}/{voc_annotation_full_file_name}", "r") as voc_annotation_file:
            tree = ET.parse(voc_annotation_file)
            root = tree.getroot()
            size = root.find("size")
            w = int(size.find("width").text)
            h = int(size.find("height").text)   

            yolo_annotation_full_file_name = f"{voc_annotation_full_file_name.split('.')[0]}.txt"
            with open(f"{PATH_TO_YOLO_ANNOTATIONS}/{yolo_annotation_full_file_name}", "w") as yolo_annotation_file:
                for obj in root.iter("object"):
                    difficult = obj.find("difficult").text
                    cls = obj.find("name").text
                    if cls not in classes or int(difficult) == 1:
                        raise(f"There are other classes beyond the license plate class: {cls}")

                    cls_id = classes.index(cls)
                    xml_box = obj.find("bndbox")
                    b = (
                    float(xml_box.find("xmin").text),
                    float(xml_box.find("xmax").text),
                    float(xml_box.find("ymin").text),
                    float(xml_box.find("ymax").text),
                    )
                    bb = convert((w, h), b)
                    print(str(cls_id) + " " + " ".join([str(a) for a in bb]), file=yolo_annotation_file)
    
    print("Done Processing")

if __name__ == "__main__":
    convert_from_voc_to_yolo()