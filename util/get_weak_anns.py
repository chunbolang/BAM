from __future__ import absolute_import, division

import networkx as nx
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, maximum_filter
from scipy.special import comb
from skimage.filters import rank
from skimage.morphology import dilation, disk, erosion, medial_axis
from sklearn.neighbors import radius_neighbors_graph
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage

def find_bbox(mask):
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    return stats[1:]  # remove bg stat

def transform_anns(mask, ann_type):
    mask_ori = mask.copy()

    if ann_type == 'bbox':
        bboxs = find_bbox(mask)
        for j in bboxs: 
            cv2.rectangle(mask, (j[0], j[1]), (j[0] + j[2], j[1] + j[3]), 1, -1) # -1->fill; 2->draw_rec        
        return mask, mask_ori
    
    elif ann_type == 'mask':
        return mask, mask_ori


if __name__ == '__main__':
    label_path = '2008_001227.png'
    mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    bboxs = find_bbox(mask)
    mask_color = cv2.imread(label_path, cv2.IMREAD_COLOR)
    for j in bboxs: 
        cv2.rectangle(mask_color, (j[0], j[1]), (j[0] + j[2], j[1] + j[3]), (0,255,0), -1)
    cv2.imwrite('bbox.png', mask_color)

    print('done')
