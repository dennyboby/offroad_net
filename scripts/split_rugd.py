rugd_train_types = [
    'park-2',
    'trail',
    'trail-3',
    'trail-4',
    'trail-6',
    'trail-9',
    'trail-10',
    'trail-11',
    'trail-12',
    'trail-14',
    'trail-15',
    'village'
]

rugd_val_types = [
    'park-8',
    'trail-5'
]

rugd_test_types = [
    'creek',
    'park-1',
    'trail-7',
    'trail-13'
]

with open('train.txt') as f: 
    train_txt = f.readlines()
temp = [line[:-1] for line in train_txt]
train_txt = [line.split('_') for line in temp]

with open('val.txt') as f: 
    val_txt = f.readlines()
temp = [line[:-1] for line in val_txt]
val_txt = [line.split('_') for line in temp]

data_txt = train_txt + val_txt

out_train_txt = []
out_val_txt = []
out_test_txt = []

for line in data_txt:
    if line[0] in rugd_train_types:
        out_train_txt.append(line[0] + '_' + line[1])
    if line[0] in rugd_val_types:
        out_val_txt.append(line[0] + '_' + line[1])
    if line[0] in rugd_test_types:
        out_test_txt.append(line[0] + '_' + line[1])

print('Total data length:', len(out_train_txt)+len(out_val_txt)+len(out_test_txt))

train_txt = open("train_1.txt", "w")
for element in out_train_txt:
    train_txt.write(element + "\n")
train_txt.close()

val_txt = open("val_1.txt", "w")
for element in out_val_txt:
    val_txt.write(element + "\n")
val_txt.close()

test_txt = open("test_1.txt", "w")
for element in out_test_txt:
    test_txt.write(element + "\n")
test_txt.close()