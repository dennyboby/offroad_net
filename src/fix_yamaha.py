import pandas as pd
import os.path as osp
import os
import shutil
import mmcv
import format_dataset as fd


def has_suffix(filename, list_suffix=None):
    ans = False
    if list_suffix is None:
        list_suffix = [".png", ".jpg"]
    for suffix in list_suffix:
        if filename.endswith(suffix):
            ans = True

    return ans


def get_files(root, list_suffix=None):
    list_files = []
    for current_dir_path, current_subdirs, current_files in os.walk(root):
        for aFile in current_files:
            if has_suffix(aFile, list_suffix):
                txt_file_path = str(os.path.join(current_dir_path, aFile))
                print(txt_file_path)
                list_files.append(txt_file_path)
    return list_files


def get_yamaha_files(data_root="yamaha_v0", list_suffix=None):
    # list_files = mmcv.scandir(osp.join(data_root, ann_dir), suffix=suffix)
    list_files = get_files(data_root, list_suffix)
    list_files = [filename for filename in list_files]
    return list_files


def create_out_dir(list_files,
                   dir_out="yamaha_v1",
                   data_root="yamaha_v0",
                   img_dir="images",
                   ann_dir="annotations",
                   inf_dir="inference_images/val",
                   split_dir="splits",
                   list_suffix=None):
    print(f"Creating yamaha_v1 dataset:")

    print(f"Making dirs: ")
    list_dirs = ["images", "annotations", "splits", "inference_images/val"]
    for dir_x in list_dirs:
        temp_dir = osp.join(dir_out, dir_x)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

    print(f"Replacing slashes:")
    list_new_files = []
    list_train = []
    list_val = []
    for f_index, filename in enumerate(list_files):
        list_split = filename.split('/')

        new_file_name = os.path.join('/'.join(list_split[:-2]), f"{list_split[-2]}_{list_split[-1]}")
        # print(f"{f_index}: {new_file_name}")
        list_new_files.append(new_file_name)

    print(f"Copy files: all RGB images to images folder; labels images to annotations folder")
    for f_index, filepath in enumerate(list_files):
        list_split = filepath.split('/')
        data_src_sub_dir = list_split[-3]
        basefile_ext = list_split[-1].split('.')
        base_filename = basefile_ext[0]
        base_ext = basefile_ext[1]
        # img_type is either rgb image or labels image
        img_type = list_split[-1]

        new_file_name = f"{list_split[-2]}_{base_filename}"
        if img_type == "rgb.jpg":
            src_path = filepath

            dest_path = os.path.join(dir_out, img_dir, f"{new_file_name}.{base_ext}")
            shutil.copy2(src_path, dest_path)

            if data_src_sub_dir == "train":
                list_train.append(new_file_name)

            elif data_src_sub_dir == "valid":
                list_val.append(new_file_name)

                dest_path = os.path.join(dir_out, inf_dir, f"{new_file_name}.{base_ext}")
                shutil.copy2(src_path, dest_path)
            else:
                print(f"Wrong sub_dir: {f_index}: {filepath}")

        elif img_type == "labels.png":
            src_path = filepath

            dest_path = os.path.join(dir_out, ann_dir, f"{new_file_name}.{base_ext}")
            shutil.copy2(src_path, dest_path)

        else:
            print(f"Wrong img_type: {f_index}: {filepath}")

    print(f"train.txt and val.txt in splits folder")
    with open(osp.join(dir_out, split_dir, "train.txt"), "w") as fh_train:
        str_lines = '\n'.join(list_train)
        fh_train.write(str_lines)

    with open(osp.join(dir_out, split_dir, "val.txt"), "w") as fh_val:
        str_lines = '\n'.join(list_val)
        fh_val.write(str_lines)


def main():
    data_root = "yamaha_v0"
    list_files = get_yamaha_files(data_root)
    create_out_dir(list_files)


if __name__ == '__main__':
    main()
