import numpy as np
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths

import pandas as pd

"""
rugd 

---annotations

---new_annotations
------gravel
------concrete
"""

# os.path.join(
#           path_till_new_ann,
#           "gravel",
#           file_name)
#

# os.makedirs(dir)

DICT_COLOR = {
    "gravel": (0, 1, 2),
    "concrete": (1, 2, 3)
}


def process_img(file, img_class):
    R, G, B = DICT_COLOR[img_class]
    is_class_present, dest_path = process_image(file, img_class, R, G, B)
    return is_class_present, dest_path


def modify_and_track(list_files):
    list_records = []
    list_interest_class = ["gravel", "concrete"]

    for index, file in enumerate(list_files):
        for img_class in list_interest_class:
            is_class_present, dest_path = process_img(file, img_class)  # for e.g. interest_class = "gravel"
            dict_rec = {
                "file": file,
                "original_class": img_class,
                "destination_path": dest_path,
                "has_interest_class": is_class_present
            }
            list_records.append(dict_rec)
    df_rec = pd.DataFrame.from_records(list_records)
    df_rec.to_csv(f"df_original_data.csv", index=False)


def class_segmentor(img, R, G, B):
    # img[:,:,2]=0 #R
    # img[:,:,1]=0 #G
    # img[:,:,0]=0 #B
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            a = img[row, col]
            if (a[0] != B or a[1] != G or a[2] != R):
                img[row, col][0] = 0
                img[row, col][1] = 0
                img[row, col][2] = 0
            else:
                img[row, col][0] = 255
                img[row, col][1] = 255
                img[row, col][2] = 255

    return img


def process_image(file, img_class, R, G, B):
    """

    """
    img = cv.imread(file)
    # class -> string: r,g,b
    # check which class is present
    # R, G, B = [101, 101, 11]
    img = class_segmentor(img, R, G, B)
    # TODO DERIVE destination path
    name = 'G:\MS Courses\Deep Learning\Group Project\my\RUGD_annotations_combined_OffroadNet\img (' + str(
        i) + ').png'
    cv.imwrite(name, img)
    # cv.imshow('img',img)
    i += 1
    # TODO add code to check whether given
    is_class_present = check_class(img)
    return is_class_present, name


def get_paths(root_dir):
    pass


if __name__ == '__main__':
    # imagePaths=list(paths.list_images("G:\MS Courses\Deep Learning\Group Project\my\\RUGD_annotations_combined"))
    root_dir = "../RUGD"
    # ["../RUGD/x.png", "../RUGD/y.png"]
    list_image_paths = get_paths(root_dir)
    modify_and_track(list_image_paths)
    # for i in range(1, 7437):
    #     process_image(i)

    # imagePaths=list(paths.list_images("G:\MS Courses\Deep Learning\Group Project\my\\RUGD_annotations_combined"))

    # i=1
    # for image in imagePaths:

    #     img=cv.imread(image)
    #     R,G,B=[102,102,0]
    #     img=class_segmentor(img,R,G,B)
    #     name='G:\MS Courses\Deep Learning\Group Project\my\RUGD_annotations_combined_OffroadNet\img ('+str(i)+').png'
    #     cv.imwrite(name,img)
    #     # cv.imshow('img',img)  
    #     i+=1
