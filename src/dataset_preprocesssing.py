import numpy as np
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# from imutils import paths
import itertools
from PIL import Image
from joblib import Parallel, delayed

import pandas as pd

"""
Goal: To re-annotate the original annotation images
for the interest_classes and 
save it under different folders

rest = [0 ,0 ,0]
interest_class = [255, 255, 255]

rugd 

---annotations

---new_annotations
------gravel
------concrete
"""

LIST_CLASSES = ["gravel", "concrete", "asphalt"]

DICT_COLOR = {
    "gravel": (255, 128, 0),
    "concrete": (101, 101, 11),
    "asphalt": (64, 64, 64)
}


# def process_img(file, img_class):
#     is_class_present, dest_path = process_image(file, img_class, R, G, B)
#     return is_class_present, dest_path


def create_combo(list_a, list_b):
    list_combo = list(itertools.product(list_a, list_b))
    return list_combo


def change_segmentation(file_name, red, green, blue, dest_path):
    img = cv.imread(file_name)
    img, is_class_present = class_segmentor(img, red, green, blue)

    # Save image only if the class is present
    if is_class_present:
        cv.imwrite(dest_path, img)
    return is_class_present


def process_image(work_dir, data):
    file_name, img_class = data
    dest_path = os.path.join(work_dir, img_class, file_name)
    red, green, blue = DICT_COLOR[img_class]
    is_class_present = change_segmentation(file_name, red, green, blue, dest_path)
    dict_rec = {
        "file_name": file_name,
        "original_class": img_class,
        "destination_path": dest_path,
        "has_interest_class": is_class_present
    }
    return dict_rec


def modify_and_track(list_files, work_dir, num_jobs=1):
    list_tuples = create_combo(list_files, LIST_CLASSES)

    result = Parallel(n_jobs=num_jobs)(delayed(process_image)(work_dir, tup_x) for tup_x in list_tuples)
    records, i = zip(*result)

    df_rec = pd.DataFrame.from_records(list(records))
    df_rec.to_csv(f"df_original_data.csv", index=False)


def get_unique(img_numpy):
    list_unique_colors = np.unique(
        img_numpy.view(np.dtype((np.void, img_numpy.dtype.itemsize * img_numpy.shape[1])))
    ).view(img_numpy.dtype).reshape(-1, img_numpy.shape[1])

    print(list_unique_colors)


def fast_color_changer():
    im = Image.open('fig1.png')
    data = np.array(im)

    r1, g1, b1 = 0, 0, 0  # Original value
    r2, g2, b2 = 255, 255, 255  # Value that we want to replace it with

    red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    data[:, :, :3][mask] = [r2, g2, b2]

    im = Image.fromarray(data)
    im.save('fig1_modified.png')


def class_segmentor(img, red, green, blue):
    list_uniq_colors = get_unique(img)
    is_class_present = False
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            a = img[row, col]

            if a[0] == blue and a[1] == green and a[2] == red:
                img[row, col][0] = 255
                img[row, col][1] = 255
                img[row, col][2] = 255
                is_class_present = True

            else:
                img[row, col][0] = 0
                img[row, col][1] = 0
                img[row, col][2] = 0

    return img, is_class_present


# def process_image(file, img_class, R, G, B):
#     """
#
#     """
#     img = cv.imread(file)
#     # class -> string: r,g,b
#     # check which class is present
#     # R, G, B = [101, 101, 11]
#     # img = class_segmentor(img, R, G, B)
#     #  DERIVE destination path
#     name = 'G:\MS Courses\Deep Learning\Group Project\my\RUGD_annotations_combined_OffroadNet\img (' + str(
#         i) + ').png'
#     cv.imwrite(name, img)
#     # cv.imshow('img',img)
#     i += 1
#     #  add code to check whether given
#     is_class_present = check_class(img)
#     return is_class_present, name


def get_paths(root_dir):
    # list_file_paths = [os.path.abspath(x) for x in os.listdir(root_dir)]
    list_file_paths = [os.path.join(root_dir, x) for x in os.listdir(root_dir)]
    return list_file_paths


def main():
    root_dir = "../RUGD/RUGD_sample-data/"
    images_path = "annotations"
    for class_x in LIST_CLASSES:
        work_dir = os.path.join(root_dir, "new_annotations", class_x)
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
    work_dir = os.path.join(root_dir, "new_annotations")
    num_jobs = 1
    list_image_paths = get_paths(os.path.join(root_dir, images_path))
    modify_and_track(list_image_paths, work_dir, num_jobs)


if __name__ == '__main__':
    main()
