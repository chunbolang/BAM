import cv2
import numpy as np
import argparse
import os.path as osp
from tqdm import tqdm
from util import get_train_val_set, check_makedirs

# Get the annotations of base categories

# root_path
# ├── BAM/
# │   ├── util/
# │   ├── config/
# │   ├── model/
# │   ├── README.md
# │   ├── train.py
# │   ├── train_base.py
# │   └── test.py
# └── data/
#     ├── base_annotation/   # the scripts to create THIS folder
#     │   ├── pascal/
#     │   │   ├── train/   
#     │   │   │   ├── 0/     # annotations of PASCAL-5^0
#     │   │   │   ├── 1/
#     │   │   │   ├── 2/
#     │   │   │   └── 3/
#     │   │   └── val/      
#     │   └── coco/          # the same file structure for COCO
#     ├── VOCdevkit2012/
#     └── MSCOCO2014/

parser = argparse.ArgumentParser()
args = parser.parse_args()

args.data_set = 'pascal'  # pascal coco
args.use_split_coco = True
args.mode = 'train'       # train val
args.split = 0            # 0 1 2 3
if args.data_set == 'pascal':
    num_classes = 20
elif args.data_set == 'coco':
    num_classes = 80

root_path = '/disk2/lcb/study/FS_Seg/'
data_path = osp.join(root_path, 'data/base_annotation/')
save_path = osp.join(data_path, args.data_set, args.mode, str(args.split))
check_makedirs(save_path)

# get class list
sub_list, sub_val_list = get_train_val_set(args)

# get data_list
fss_list_root = root_path + '/BAM/lists/{}/fss_list/{}/'.format(args.data_set, args.mode)
fss_data_list_path = fss_list_root + 'data_list_{}.txt'.format(args.split)
with open(fss_data_list_path, 'r') as f:
    f_str = f.readlines()
data_list = []
for line in f_str:
    img, mask = line.split(' ')
    data_list.append((img, mask.strip()))

# Start Processing
for index in tqdm(range(len(data_list))):
    image_path, label_path = data_list[index]
    image_path, label_path = root_path + image_path[3:], root_path+ label_path[3:]  # 
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    label_tmp = label.copy()

    for cls in range(1,num_classes+1):
        select_pix = np.where(label_tmp == cls)
        if cls in sub_list:
            label[select_pix[0],select_pix[1]] = sub_list.index(cls) + 1
        else:
            label[select_pix[0],select_pix[1]] = 0

    # for pix in np.nditer(label, op_flags=['readwrite']):
    #     if pix == 255:
    #         pass
    #     elif pix not in sub_list: 
    #         pix[...] = 0
    #     else:
    #         pix[...] = sub_list.index(pix) + 1
    
    save_item_path = osp.join(save_path, label_path.split('/')[-1])
    cv2.imwrite(save_item_path, label)


print('end')
