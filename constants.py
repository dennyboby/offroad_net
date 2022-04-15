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

# import pandas as pd
#
# df = pd.read_csv(f"RUGD/RUGD_sample-data/RUGD_annotation-colormap.txt", sep=" ")
# print(df)
# df.columns = ["sr_no", "class", "R", "G", "B"]
# print(df)
# print(list(df["class"]))
# for index, row in df.iterrows():
#     print(f"[{row['R']},{row['G']},{row['B']}],")
