import os
import os.path
import cv2
import numpy as np
import copy

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random
import time
from tqdm import tqdm

from .get_weak_anns import transform_anns

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']



def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split=0, data_root=None, data_list=None, sub_list=None, filter_intersection=False):    
    assert split in [0, 1, 2, 3]
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    # Shaban uses these lines to remove small objects:
    # if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
    #    filtered_item.append(item)
    # which means the mask will be downsampled to 1/32 of the original size and the valid area should be larger than 2, 
    # therefore the area in original size should be accordingly larger than 2 * 32 * 32    
    image_label_list = []  
    list_read = open(data_list).readlines()
    print("Processing data...".format(sub_list))
    sub_class_file_list = {}
    for sub_c in sub_list:
        sub_class_file_list[sub_c] = []

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split(' ')
        image_name = os.path.join(data_root, line_split[0])
        label_name = os.path.join(data_root, line_split[1])
        item = (image_name, label_name)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        label_class = np.unique(label).tolist()

        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)

        new_label_class = []     

        if filter_intersection:  # filter images containing objects of novel categories during meta-training
            if set(label_class).issubset(set(sub_list)):
                for c in label_class:
                    if c in sub_list:
                        tmp_label = np.zeros_like(label)
                        target_pix = np.where(label == c)
                        tmp_label[target_pix[0],target_pix[1]] = 1 
                        if tmp_label.sum() >= 2 * 32 * 32:      
                            new_label_class.append(c)     
        else:
            for c in label_class:
                if c in sub_list:
                    tmp_label = np.zeros_like(label)
                    target_pix = np.where(label == c)
                    tmp_label[target_pix[0],target_pix[1]] = 1 
                    if tmp_label.sum() >= 2 * 32 * 32:      
                        new_label_class.append(c)            

        label_class = new_label_class

        if len(label_class) > 0:
            image_label_list.append(item)
            for c in label_class:
                if c in sub_list:
                    sub_class_file_list[c].append(item)
                    
    print("Checking image&label pair {} list done! ".format(split))
    return image_label_list, sub_class_file_list



class SemData(Dataset):
    def __init__(self, split=3, shot=1, data_root=None, base_data_root=None, data_list=None, data_set=None, use_split_coco=False, \
                        transform=None, transform_tri=None, mode='train', ann_type='mask', \
                        ft_transform=None, ft_aug_size=None, \
                        ms_transform=None):

        assert mode in ['train', 'val', 'demo', 'finetune']
        assert data_set in ['pascal', 'coco']
        if mode == 'finetune':
            assert ft_transform is not None
            assert ft_aug_size is not None

        if data_set == 'pascal':
            self.num_classes = 20
        elif data_set == 'coco':
            self.num_classes = 80

        self.mode = mode
        self.split = split  
        self.shot = shot
        self.data_root = data_root   
        self.base_data_root = base_data_root   
        self.ann_type = ann_type

        if data_set == 'pascal':
            self.class_list = list(range(1, 21))                         # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            if self.split == 3: 
                self.sub_list = list(range(1, 16))                       # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                self.sub_val_list = list(range(16, 21))                  # [16,17,18,19,20]
            elif self.split == 2:
                self.sub_list = list(range(1, 11)) + list(range(16, 21)) # [1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
                self.sub_val_list = list(range(11, 16))                  # [11,12,13,14,15]
            elif self.split == 1:
                self.sub_list = list(range(1, 6)) + list(range(11, 21))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(6, 11))                   # [6,7,8,9,10]
            elif self.split == 0:
                self.sub_list = list(range(6, 21))                       # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(1, 6))                    # [1,2,3,4,5]

        elif data_set == 'coco':
            if use_split_coco:
                print('INFO: using SPLIT COCO (FWB)')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_val_list = list(range(4, 81, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))                    
                elif self.split == 2:
                    self.sub_val_list = list(range(3, 80, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
                elif self.split == 1:
                    self.sub_val_list = list(range(2, 79, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
                elif self.split == 0:
                    self.sub_val_list = list(range(1, 78, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
            else:
                print('INFO: using COCO (PANet)')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_list = list(range(1, 61))
                    self.sub_val_list = list(range(61, 81))
                elif self.split == 2:
                    self.sub_list = list(range(1, 41)) + list(range(61, 81))
                    self.sub_val_list = list(range(41, 61))
                elif self.split == 1:
                    self.sub_list = list(range(1, 21)) + list(range(41, 81))
                    self.sub_val_list = list(range(21, 41))
                elif self.split == 0:
                    self.sub_list = list(range(21, 81)) 
                    self.sub_val_list = list(range(1, 21))           

        print('sub_list: ', self.sub_list)
        print('sub_val_list: ', self.sub_val_list)    

        # @@@ For convenience, we skip the step of building datasets and instead use the pre-generated lists @@@
        # if self.mode == 'train':
        #     self.data_list, self.sub_class_file_list = make_dataset(split, data_root, data_list, self.sub_list, True)
        #     assert len(self.sub_class_file_list.keys()) == len(self.sub_list)
        # elif self.mode == 'val' or self.mode == 'demo' or self.mode == 'finetune':
        #     self.data_list, self.sub_class_file_list = make_dataset(split, data_root, data_list, self.sub_val_list, False)
        #     assert len(self.sub_class_file_list.keys()) == len(self.sub_val_list) 

        mode = 'train' if self.mode=='train' else 'val'
        self.base_path = os.path.join(self.base_data_root, mode, str(self.split))

        fss_list_root = './lists/{}/fss_list/{}/'.format(data_set, mode)
        fss_data_list_path = fss_list_root + 'data_list_{}.txt'.format(split)
        fss_sub_class_file_list_path = fss_list_root + 'sub_class_file_list_{}.txt'.format(split)

        # Write FSS Data
        # with open(fss_data_list_path, 'w') as f:
        #     for item in self.data_list:
        #         img, label = item
        #         f.write(img + ' ')
        #         f.write(label + '\n')
        # with open(fss_sub_class_file_list_path, 'w') as f:
        #     f.write(str(self.sub_class_file_list))

        # Read FSS Data
        with open(fss_data_list_path, 'r') as f:
            f_str = f.readlines()
        self.data_list = []
        for line in f_str:
            img, mask = line.split(' ')
            self.data_list.append((img, mask.strip()))

        with open(fss_sub_class_file_list_path, 'r') as f:
            f_str = f.read()
        self.sub_class_file_list = eval(f_str)

        self.transform = transform
        self.transform_tri = transform_tri
        self.ft_transform = ft_transform
        self.ft_aug_size = ft_aug_size
        self.ms_transform_list = ms_transform
      
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        label_class = []
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label_b = cv2.imread(os.path.join(self.base_path,label_path.split('/')[-1]), cv2.IMREAD_GRAYSCALE)

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))          
        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255) 
        new_label_class = []
        for c in label_class:
            if c in self.sub_val_list:
                if self.mode == 'val' or self.mode == 'demo' or self.mode == 'finetune':
                    new_label_class.append(c)
            if c in self.sub_list:
                if self.mode == 'train':
                    new_label_class.append(c)
        label_class = new_label_class    
        assert len(label_class) > 0

        class_chosen = label_class[random.randint(1,len(label_class))-1]
        target_pix = np.where(label == class_chosen)
        ignore_pix = np.where(label == 255)
        label[:,:] = 0
        if target_pix[0].shape[0] > 0:
            label[target_pix[0],target_pix[1]] = 1 
        label[ignore_pix[0],ignore_pix[1]] = 255

        # for cls in range(1,self.num_classes+1):
        #     select_pix = np.where(label_b_tmp == cls)
        #     if cls in self.sub_list:
        #         label_b[select_pix[0],select_pix[1]] = self.sub_list.index(cls) + 1
        #     else:
        #         label_b[select_pix[0],select_pix[1]] = 0    

        file_class_chosen = self.sub_class_file_list[class_chosen]
        num_file = len(file_class_chosen)

        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []
        for k in range(self.shot):
            support_idx = random.randint(1,num_file)-1
            support_image_path = image_path
            support_label_path = label_path
            while((support_image_path == image_path and support_label_path == label_path) or support_idx in support_idx_list):
                support_idx = random.randint(1,num_file)-1
                support_image_path, support_label_path = file_class_chosen[support_idx]                
            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)

        support_image_list_ori = []
        support_label_list_ori = []
        support_label_list_ori_mask = []
        subcls_list = []
        if self.mode == 'train':
            subcls_list.append(self.sub_list.index(class_chosen))
        else:
            subcls_list.append(self.sub_val_list.index(class_chosen))        
        for k in range(self.shot):  
            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k] 
            support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)      
            support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
            support_image = np.float32(support_image)
            support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
            target_pix = np.where(support_label == class_chosen)
            ignore_pix = np.where(support_label == 255)
            support_label[:,:] = 0
            support_label[target_pix[0],target_pix[1]] = 1 
            
            support_label, support_label_mask = transform_anns(support_label, self.ann_type)   # mask/bbox
            support_label[ignore_pix[0],ignore_pix[1]] = 255
            support_label_mask[ignore_pix[0],ignore_pix[1]] = 255
            if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
                raise (RuntimeError("Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))            
            support_image_list_ori.append(support_image)
            support_label_list_ori.append(support_label)
            support_label_list_ori_mask.append(support_label_mask)
        assert len(support_label_list_ori) == self.shot and len(support_image_list_ori) == self.shot                    
        
        raw_image = image.copy()
        raw_label = label.copy()
        raw_label_b = label_b.copy()
        support_image_list = [[] for _ in range(self.shot)]
        support_label_list = [[] for _ in range(self.shot)]
        if self.transform is not None:
            image, label, label_b = self.transform_tri(image, label, label_b)   # transform the triple
            for k in range(self.shot):
                support_image_list[k], support_label_list[k] = self.transform(support_image_list_ori[k], support_label_list_ori[k])

        s_xs = support_image_list
        s_ys = support_label_list
        s_x = s_xs[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
        s_y = s_ys[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)

        # Return
        if self.mode == 'train':
            return image, label, label_b, s_x, s_y, subcls_list
        elif self.mode == 'val':
            return image, label, label_b, s_x, s_y, subcls_list, raw_label, raw_label_b
        elif self.mode == 'demo':
            total_image_list = support_image_list_ori.copy()
            total_image_list.append(raw_image)            
            return image, label, label_b, s_x, s_y, subcls_list, total_image_list, support_label_list_ori, support_label_list_ori_mask, raw_label, raw_label_b          



# -------------------------- GFSS --------------------------

def make_GFSS_dataset(split=0, data_root=None, data_list=None, sub_list=None, sub_val_list=None):
    assert split in [0, 1, 2, 3]
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    image_label_list = []  
    list_read = open(data_list).readlines()
    print("Processing data...".format(sub_val_list))
    sub_class_list_sup = {}
    for sub_c in sub_val_list:
        sub_class_list_sup[sub_c] = []

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split(' ')
        image_name = os.path.join(data_root, line_split[0])
        label_name = os.path.join(data_root, line_split[1])
        item = (image_name, label_name)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        label_class = np.unique(label).tolist()

        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)

        for c in label_class:
            if c in sub_val_list:
                sub_class_list_sup[c].append(item)
       
        image_label_list.append(item)
    
    print("Checking image&label pair {} list done! ".format(split))
    return sub_class_list_sup, image_label_list

class GSemData(Dataset):
    # Generalized Few-Shot Segmentation    
    def __init__(self, split=3, shot=1, data_root=None, base_data_root=None, data_list=None, data_set=None, use_split_coco=False, \
                        transform=None, transform_tri=None, mode='val', ann_type='mask'):

        assert mode in ['val', 'demo']
        assert data_set in ['pascal', 'coco']

        if data_set == 'pascal':
            self.num_classes = 20
        elif data_set == 'coco':
            self.num_classes = 80

        self.mode = mode
        self.split = split  
        self.shot = shot
        self.data_root = data_root   
        self.base_data_root = base_data_root   
        self.ann_type = ann_type

        if data_set == 'pascal':
            self.class_list = list(range(1, 21))                         # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            if self.split == 3: 
                self.sub_list = list(range(1, 16))                       # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                self.sub_val_list = list(range(16, 21))                  # [16,17,18,19,20]
            elif self.split == 2:
                self.sub_list = list(range(1, 11)) + list(range(16, 21)) # [1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
                self.sub_val_list = list(range(11, 16))                  # [11,12,13,14,15]
            elif self.split == 1:
                self.sub_list = list(range(1, 6)) + list(range(11, 21))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(6, 11))                   # [6,7,8,9,10]
            elif self.split == 0:
                self.sub_list = list(range(6, 21))                       # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(1, 6))                    # [1,2,3,4,5]

        elif data_set == 'coco':
            if use_split_coco:
                print('INFO: using SPLIT COCO (FWB)')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_val_list = list(range(4, 81, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))                    
                elif self.split == 2:
                    self.sub_val_list = list(range(3, 80, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
                elif self.split == 1:
                    self.sub_val_list = list(range(2, 79, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
                elif self.split == 0:
                    self.sub_val_list = list(range(1, 78, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
            else:
                print('INFO: using COCO (PANet)')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_list = list(range(1, 61))
                    self.sub_val_list = list(range(61, 81))
                elif self.split == 2:
                    self.sub_list = list(range(1, 41)) + list(range(61, 81))
                    self.sub_val_list = list(range(41, 61))
                elif self.split == 1:
                    self.sub_list = list(range(1, 21)) + list(range(41, 81))
                    self.sub_val_list = list(range(21, 41))
                elif self.split == 0:
                    self.sub_list = list(range(21, 81)) 
                    self.sub_val_list = list(range(1, 21))           

        print('sub_list: ', self.sub_list)
        print('sub_val_list: ', self.sub_val_list)    

        self.sub_class_list_sup, self.data_list = make_GFSS_dataset(split, data_root, data_list, self.sub_list, self.sub_val_list)
        assert len(self.sub_class_list_sup.keys()) == len(self.sub_val_list) 

        self.transform = transform
        self.transform_tri = transform_tri


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        # Choose a query image
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label_t = label.copy()
        label_t_tmp = label.copy()
        
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))       

        # Get the category information of the query image
        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255) 
        label_class_novel = []
        label_class_base = []
        for c in label_class:
            if c in self.sub_val_list:
                label_class_novel.append(c)
            else:
                label_class_base.append(c)

        # Choose the category of this episode
        if len(label_class_base) == 0:
            class_chosen = random.choice(label_class_novel) # rule out the possibility that the image contains only "background"
        else:
            class_chosen = random.choice(self.sub_val_list)

        # Generate new annotations
        for cls in range(1,self.num_classes+1):
            select_pix = np.where(label_t_tmp == cls)
            if cls in self.sub_list:
                label_t[select_pix[0],select_pix[1]] = self.sub_list.index(cls) + 1
            elif cls == class_chosen:
                label_t[select_pix[0],select_pix[1]] = self.num_classes*3/4 + 1
            else:
                label_t[select_pix[0],select_pix[1]] = 0  

        # Sample K-shot images
        file_class_chosen = self.sub_class_list_sup[class_chosen]
        num_file = len(file_class_chosen)

        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []
        for k in range(self.shot):
            support_idx = random.randint(1,num_file)-1
            support_image_path = image_path
            support_label_path = label_path
            while((support_image_path == image_path and support_label_path == label_path) or support_idx in support_idx_list):
                support_idx = random.randint(1,num_file)-1
                support_image_path, support_label_path = file_class_chosen[support_idx]                
            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)

        support_image_list_ori = []
        support_label_list_ori = []
        support_label_list_ori_mask = []
        subcls_list = []
        subcls_list.append(self.sub_val_list.index(class_chosen))      
        for k in range(self.shot):  
            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k] 
            support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)      
            support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
            support_image = np.float32(support_image)
            support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
            target_pix = np.where(support_label == class_chosen)
            ignore_pix = np.where(support_label == 255)
            support_label[:,:] = 0
            support_label[target_pix[0],target_pix[1]] = 1 
            
            support_label, support_label_mask = transform_anns(support_label, self.ann_type)
            support_label[ignore_pix[0],ignore_pix[1]] = 255
            support_label_mask[ignore_pix[0],ignore_pix[1]] = 255
            if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
                raise (RuntimeError("Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))            
            support_image_list_ori.append(support_image)
            support_label_list_ori.append(support_label)
            support_label_list_ori_mask.append(support_label_mask)
        assert len(support_label_list_ori) == self.shot and len(support_image_list_ori) == self.shot             
        
        # Transform
        raw_image = image.copy()
        raw_label_t = label_t.copy()
        support_image_list = [[] for _ in range(self.shot)]
        support_label_list = [[] for _ in range(self.shot)]
        if self.transform is not None:
            image, label_t = self.transform(image, label_t)
            for k in range(self.shot):
                support_image_list[k], support_label_list[k] = self.transform(support_image_list_ori[k], support_label_list_ori[k])

        s_xs = support_image_list
        s_ys = support_label_list
        s_x = s_xs[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
        s_y = s_ys[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)

        # Return
        if self.mode == 'val':
            return image, label_t, s_x, s_y, subcls_list, raw_label_t
        elif self.mode == 'demo':
            total_image_list = support_image_list_ori.copy()
            total_image_list.append(raw_image)            
            return image, label_t, s_x, s_y, subcls_list, total_image_list, support_label_list_ori, support_label_list_ori_mask, raw_label_t 



# -------------------------- Pre-Training --------------------------

class BaseData(Dataset):
    def __init__(self, split=3, mode=None, data_root=None, data_list=None, data_set=None, use_split_coco=False, transform=None, main_process=False, \
                batch_size=None):

        assert data_set in ['pascal', 'coco']
        assert mode in ['train', 'val']

        if data_set == 'pascal':
            self.num_classes = 20
        elif data_set == 'coco':
            self.num_classes = 80

        self.mode = mode
        self.split = split 
        self.data_root = data_root
        self.batch_size = batch_size

        if data_set == 'pascal':
            self.class_list = list(range(1, 21))                         # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            if self.split == 3: 
                self.sub_list = list(range(1, 16))                       # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                self.sub_val_list = list(range(16, 21))                  # [16,17,18,19,20]
            elif self.split == 2:
                self.sub_list = list(range(1, 11)) + list(range(16, 21)) # [1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
                self.sub_val_list = list(range(11, 16))                  # [11,12,13,14,15]
            elif self.split == 1:
                self.sub_list = list(range(1, 6)) + list(range(11, 21))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(6, 11))                   # [6,7,8,9,10]
            elif self.split == 0:
                self.sub_list = list(range(6, 21))                       # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(1, 6))                    # [1,2,3,4,5]

        elif data_set == 'coco':
            if use_split_coco:
                print('INFO: using SPLIT COCO (FWB)')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_val_list = list(range(4, 81, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))                    
                elif self.split == 2:
                    self.sub_val_list = list(range(3, 80, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
                elif self.split == 1:
                    self.sub_val_list = list(range(2, 79, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
                elif self.split == 0:
                    self.sub_val_list = list(range(1, 78, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
            else:
                print('INFO: using COCO (PANet)')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_list = list(range(1, 61))
                    self.sub_val_list = list(range(61, 81))
                elif self.split == 2:
                    self.sub_list = list(range(1, 41)) + list(range(61, 81))
                    self.sub_val_list = list(range(41, 61))
                elif self.split == 1:
                    self.sub_list = list(range(1, 21)) + list(range(41, 81))
                    self.sub_val_list = list(range(21, 41))
                elif self.split == 0:
                    self.sub_list = list(range(21, 81)) 
                    self.sub_val_list = list(range(1, 21))           

        print('sub_list: ', self.sub_list)
        print('sub_val_list: ', self.sub_val_list)    

        self.data_list = []  
        list_read = open(data_list).readlines()
        print("Processing data...")

        for l_idx in tqdm(range(len(list_read))):
            line = list_read[l_idx]
            line = line.strip()
            line_split = line.split(' ')
            image_name = os.path.join(self.data_root, line_split[0])
            label_name = os.path.join(self.data_root, line_split[1])
            item = (image_name, label_name)
            self.data_list.append(item)

        self.transform = transform


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label_tmp = label.copy()

        for cls in range(1, self.num_classes+1):
            select_pix = np.where(label_tmp == cls)
            if cls in self.sub_list:
                label[select_pix[0],select_pix[1]] = self.sub_list.index(cls) + 1
            else:
                label[select_pix[0],select_pix[1]] = 0
                
        raw_label = label.copy()

        if self.transform is not None:
            image, label = self.transform(image, label)

        # Return
        if self.mode == 'val' and self.batch_size == 1:
            return image, label, raw_label
        else:
            return image, label
      