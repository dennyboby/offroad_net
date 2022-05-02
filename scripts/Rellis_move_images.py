import glob
import shutil
import os

address0="Rellis_3D_image_example/"
src_dir = ['00000','00001','00002','00003','00004']
address1="/pylon_camera_node"
address2="Rellis-3D 2/"
address3="/pylon_camera_node_label_color" 
dst_dir = "images"
dst_dir2 = "annotations"

with open('Rellis_3D_image_split/test_t.lst') as f: 
    test_txt = f.readlines()
test_txt = [line.replace('\n','').split(" ") for line in test_txt]

for folder in src_dir:
    address_img=address0+folder+address1
    for jpgfile in glob.iglob(os.path.join(address_img, "*.jpg")):
        image_name = jpgfile.replace("Rellis_3D_image_example/0","0")
        image_name=image_name.replace("\\","/") 
        for i in range(len(test_txt)):
            if image_name in test_txt[i][0]:            
                shutil.copy(jpgfile, dst_dir)

    address_ano=address2+folder+address3

    for jpgfile in glob.iglob(os.path.join(address_ano, "*.png")):
        image_name = jpgfile.replace("Rellis-3D 2/0","0")
        image_name=image_name.replace("\\","/") 
        for i in range(len(test_txt)):
            if image_name in test_txt[i][1]:            
                shutil.copy(jpgfile, dst_dir2)
