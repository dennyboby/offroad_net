import pandas as pd
import os.path as osp
import os
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


def get_yamaha_files(data_root="yamaha_v0", ann_dir="", split_dir="splits", list_suffix=None):
    # list_files = mmcv.scandir(osp.join(data_root, ann_dir), suffix=suffix)
    list_files = get_files(data_root, list_suffix)
    list_files = [filename for filename in list_files]
    return list_files


def create_out_dir(list_files, dir_out="yamaha_v1", data_root="yamaha_v0", ann_dir="", split_dir="splits",
                   suffix=".png"):
    print(f"Creating yamaha_v1 dataset:")

    print(f"Making dirs: ")
    list_dirs = ["images", "annotations", "splits"]
    for dir in list_dirs:
        mmcv.mkdir_or_exist(osp.join(dir_out, dir))

    print(f"Replacing slashes:")
    list_new_files = []
    list_train = []
    list_val = []
    for f_index, filename in enumerate(list_files):
        list_split = filename.split('/')

        data_src_sub_dir = list_split[-3]
        list_fn_split = list_split[-1].split('.')
        file_id = list_fn_split[0]
        file_ext = list_fn_split[1]
        if data_src_sub_dir == "train":
            list_train.append(file_id)
        elif data_src_sub_dir == "valid":
            list_val.append(file_id)
        else:
            print(f"Wrong sub_dir: {f_index}: {filename}")

        new_file_name = os.path.join('/'.join(list_split[:-2]), f"{list_split[-2]}_{list_split[-1]}")
        # print(f"{f_index}: {new_file_name}")
        list_new_files.append(new_file_name)

    print(f"Move to all RGB images to images folder and labels images to annotations folder")
    for f_index, filename in enumerate(list_new_files):
        sub_dir = filename.split()

    print(f"train.txt and val.txt in splits folder")
    with open(osp.join(split_dir, "train.txt"), "w") as fh_train:
        fh_train.writelines(list_train)

    with open(osp.join(split_dir, "val.txt"), "w") as fh_val:
        fh_val.writelines(list_val)


def main():
    data_root = "temp_yamaha"
    list_files = get_yamaha_files(data_root)
    create_out_dir(list_files)


if __name__ == '__main__':
    main()
