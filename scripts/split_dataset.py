import os
import random
import shutil


IMGS_BASE_PATH = "../5-final-dataset/images"
LABELS_BASE_PATH = "../5-final-dataset/labels"

def make_train_validation_test_split(fp_images, fp_labels, train_ratio, validation_ratio, test_ratio):
    img_full_names = os.listdir(fp_images)
    number_of_images = len(img_full_names)
    random.shuffle(img_full_names)

    number_of_images_in_training_set = int(number_of_images * train_ratio)
    number_of_images_in_validation_set = int(number_of_images * validation_ratio)
    # number_of_images_in_test_set = number_of_images * test_ratio

    training_index = number_of_images_in_training_set
    validation_index = number_of_images_in_validation_set + training_index
    # test_index = number_of_images_in_test_set + validation_index

    training_set_img_full_names = img_full_names[:training_index]
    validation_set_img_full_names = img_full_names[training_index:validation_index]
    test_set_img_full_names = img_full_names[validation_index:]

    print("Start Processing...")

    for img_full_name in training_set_img_full_names:
        print(f"Processing: {img_full_name}")
        shutil.copyfile(f"{fp_images}/{img_full_name}", f"{IMGS_BASE_PATH}/train/{img_full_name}")
        img_name, _ = os.path.splitext(img_full_name)
        shutil.copyfile(f"{fp_labels}/{img_name}.txt", f"{LABELS_BASE_PATH}/train/{img_name}.txt")
        print(f"Done Processing: {img_full_name}")
    for img_full_name in validation_set_img_full_names:
        print(f"Processing: {img_full_name}")
        shutil.copyfile(f"{fp_images}/{img_full_name}", f"{IMGS_BASE_PATH}/valid/{img_full_name}")
        img_name, _ = os.path.splitext(img_full_name)
        shutil.copyfile(f"{fp_labels}/{img_name}.txt", f"{LABELS_BASE_PATH}/valid/{img_name}.txt")
        print(f"Done Processing: {img_full_name}")
    for img_full_name in test_set_img_full_names:
        print(f"Processing: {img_full_name}")
        shutil.copyfile(f"{fp_images}/{img_full_name}", f"{IMGS_BASE_PATH}/test/{img_full_name}")
        img_name, _ = os.path.splitext(img_full_name)
        shutil.copyfile(f"{fp_labels}/{img_name}.txt", f"{LABELS_BASE_PATH}/test/{img_name}.txt")
        print(f"Done Processing: {img_full_name}")

    print("Done.")

if __name__ == "__main__":
    make_train_validation_test_split("./images", "./annotations", 0.7, 0.2, 0.1)
