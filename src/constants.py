rugd_dir = "RUGD/RUGD_sample-data"
# rugd_dir = "RUGD"

rugd_classes = (
    'void', 'dirt', 'sand', 'grass', 'tree',
    'pole', 'water', 'sky', 'vehicle', 'container/generic-object',
    'asphalt', 'gravel', 'building', 'mulch', 'rock-bed',
    'log', 'bicycle', 'person', 'fence', 'bush',
    'sign', 'rock', 'bridge', 'concrete', 'picnic-table'
)

rugd_palette = [
    [0, 0, 0],
    [108, 64, 20],
    [255, 229, 204],
    [0, 102, 0],
    [0, 255, 0],
    [0, 153, 153],
    [0, 128, 255],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 127],
    [64, 64, 64],
    [255, 128, 0],
    [255, 0, 0],
    [153, 76, 0],
    [102, 102, 0],
    [102, 0, 0],
    [0, 255, 128],
    [204, 153, 255],
    [102, 0, 204],
    [255, 153, 204],
    [0, 102, 102],
    [153, 204, 255],
    [102, 255, 255],
    [101, 101, 11],
    [114, 85, 47]
]

offroad_classes = (
    'background',
    'path'
)

offroad_palette = [
    [0, 0, 0],
    [255, 255, 255]
]

yamaha_classes = (
    "non-traversable low vegetation"
    "sky",
    "high vegetation",
    "traversable grass",
    "rough trail",
    "smooth trail",
    "obstacle",
    "truck"
)

yamaha_palette = [
    [0, 160, 0],
    [1, 88, 255],
    [40, 80, 0],
    [128, 255, 0],
    [156, 76, 30],
    [178, 176, 153],
    [255, 0, 0],
    [255, 255, 255]
]

rellis3d_classes = (
    "non-traversable low vegetation"
    "sky",
    "high vegetation",
    "traversable grass",
    "rough trail",
    "smooth trail",
    "obstacle",
    "truck"
)

rellis3d_palette = [
    [0, 160, 0],
    [1, 88, 255],
    [40, 80, 0],
    [128, 255, 0],
    [156, 76, 30],
    [178, 176, 153],
    [255, 0, 0],
    [255, 255, 255]
]

# import pandas as pd
#
# df = pd.read_csv(f"RUGD/RUGD_sample-data/RUGD_annotation-colormap.txt", sep=" ")
# print(df)
# df.columns = ["sr_no", "class", "R", "G", "B"]
# print(df)
# print(list(df["class"]))
# for index, row in df.iterrows():
#     print(f"[{row['R']},{row['G']},{row['B']}],")
