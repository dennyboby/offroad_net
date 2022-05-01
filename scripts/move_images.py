import glob
import shutil
import os

src_dir = "images"
dst_dir = "inference_images/test_1"

with open('splits/test.txt') as f: 
    test_txt = f.readlines()
test_txt = [line[:-1] for line in test_txt]

for jpgfile in glob.iglob(os.path.join(src_dir, "*.png")):
    image_name = jpgfile.split('/')[-1][:-4]
    if image_name in test_txt:
        shutil.copy(jpgfile, dst_dir)