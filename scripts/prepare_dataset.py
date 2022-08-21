import torch
import os
from enum import Enum
from PIL import Image
from datetime import datetime

DATASETS = [
    "1-car-plate-detection", 
    "2-dataset_sample_584", 
    "3-endtoend(lpd)", 
    "4-artificial-mercosur-license-plates", 
    "5-license-plates-dataset", 
    "6-romanian-license-plate-dataset",
]


class VehicleID(Enum):
    CAR = 2
    BUS = 5
    TRUCK = 7


def get_yolo_annotation(annotation):
    xywh = annotation.split(" ")
    x, y, w, h = float(xywh[1]), float(xywh[2]), float(xywh[3]), float(xywh[4])
    return (x, y, w, h)


def convert_yolo_to_integer(width, height, license_xywh):
    x, y, w, h = license_xywh
    xmax = int((x * width) + (w * width) / 2.0)
    xmin = int((x * width) - (w * width) / 2.0)
    ymax = int((y * height) + (h * height) / 2.0)
    ymin = int((y * height) - (h * height) / 2.0)
    return (xmin, ymin, xmax, ymax)

def convert_integer_to_yolo(width, height, license_xyxy):
    dw = 1.0 / width
    dh = 1.0 / height
    x = (license_xyxy[0] + license_xyxy[2]) / 2.0
    y = (license_xyxy[1] + license_xyxy[3]) / 2.0
    w = license_xyxy[2] - license_xyxy[0]
    h = license_xyxy[3] - license_xyxy[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def is_license_plate_inside_car(license_xyxy, car_xyxy):
    return (car_xyxy[0] <= license_xyxy[0] and car_xyxy[1] <= license_xyxy[1]) and (
        car_xyxy[2] >= license_xyxy[2] and car_xyxy[3] >= license_xyxy[3]
    )


def resize_license_plate_annotation_wrt_car_annotation(license_xyxy, car_xyxy):
    resized_license = (
        int(license_xyxy[0] - car_xyxy[0]),
        int(license_xyxy[1] - car_xyxy[1]),
        int(license_xyxy[2] - car_xyxy[0]),
        int(license_xyxy[3] - car_xyxy[1]),
    )

    return resized_license


# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5x", force_reload=True)  # or yolov5m, yolov5l, yolov5x, custom
model.classes = [
    VehicleID.CAR.value,
    # VehicleID.BUS.value,
    # VehicleID.TRUCK.value,
]

print("Start Processing...")

start_time = datetime.now()

number_of_processed_images = 0
number_of_saved_images = 0

for dataset_name in DATASETS:
    for image_full_name in os.listdir(f"./{dataset_name}/dataset/images"):
        print(f"Processing: ./{dataset_name}/dataset/images/{image_full_name}")
        # Images
        # img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list
        img = Image.open(f"./{dataset_name}/dataset/images/{image_full_name}")
        number_of_processed_images = number_of_processed_images + 1
        img_width, img_height = img.size
        img_name, _ = os.path.splitext(image_full_name)

        # Inference
        predictions = model(img)

        # Results
        # results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

        predicted_cars = predictions.pandas().xyxy[0]

        for _, car in predicted_cars.iterrows():
            
            car_xyxy = (car.xmin, car.ymin, car.xmax, car.ymax)

            with open(f"./{dataset_name}/dataset/annotations/{img_name}.txt", "r") as annotations:
                for annotation in annotations:
                    yolo_license_plate_annotation = get_yolo_annotation(annotation)
                    integer_license_plate_annotation = convert_yolo_to_integer(img_width, img_height, yolo_license_plate_annotation)
                    if is_license_plate_inside_car(integer_license_plate_annotation, car_xyxy):
                        new_integer_license_plate_annotation = resize_license_plate_annotation_wrt_car_annotation(integer_license_plate_annotation, car_xyxy)
                        car_img = img.crop(car_xyxy).convert("RGB")
                        car_img_width, car_img_height = car_img.size
                        new_yolo_license_plate_annotation = convert_integer_to_yolo(car_img_width, car_img_height, new_integer_license_plate_annotation)
                        car_img.save(f"../4-cropped-dataset/images/{number_of_saved_images}.png")
                        with open(f"../4-cropped-dataset/annotations/{number_of_saved_images}.txt", "w") as new_annotation:
                            print("0" + " " + " ".join([str(a) for a in new_yolo_license_plate_annotation]), file=new_annotation)
                        number_of_saved_images = number_of_saved_images + 1
        print(f"Done Processing: ./{dataset_name}/dataset/images/{image_full_name}")
    print(f"Processed Images: {number_of_processed_images}")
    print(f"Saved Images: {number_of_saved_images}")
print("Done.")

print(f"Elapsed time: {datetime.now() - start_time}")
