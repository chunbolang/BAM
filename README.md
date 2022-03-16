[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-what-not-to-segment-a-new/few-shot-semantic-segmentation-on-pascal-5i-1)](https://paperswithcode.com/sota/few-shot-semantic-segmentation-on-pascal-5i-1?p=learning-what-not-to-segment-a-new)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-what-not-to-segment-a-new/few-shot-semantic-segmentation-on-pascal-5i-5)](https://paperswithcode.com/sota/few-shot-semantic-segmentation-on-pascal-5i-5?p=learning-what-not-to-segment-a-new)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-what-not-to-segment-a-new/few-shot-semantic-segmentation-on-coco-20i-1)](https://paperswithcode.com/sota/few-shot-semantic-segmentation-on-coco-20i-1?p=learning-what-not-to-segment-a-new)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-what-not-to-segment-a-new/few-shot-semantic-segmentation-on-coco-20i-5)](https://paperswithcode.com/sota/few-shot-semantic-segmentation-on-coco-20i-5?p=learning-what-not-to-segment-a-new)
# Learning What Not to Segment: A New Perspective on Few-Shot Segmentation

This repo contains the code for our **CVPR 2022** [paper](http://arxiv.org/abs/2203.07615) "*Learning What Not to Segment: A New Perspective on Few-Shot Segmentation*" by Chunbo Lang, Gong Cheng, Binfei Tu, and Junwei Han. 

<p align="middle">
  <img src="figure/flowchart.jpg">
</p>

### Dependencies

- Python 3.8
- PyTorch 1.7.0
- cuda 11.0
- torchvision 0.8.1
- tensorboardX 2.14

### Datasets

- PASCAL-5<sup>i</sup>:  [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) + [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)

- COCO-20<sup>i</sup>:  [COCO2014](https://cocodataset.org/#download)

   Download the [data](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/langchunbo_mail_nwpu_edu_cn/EZboVV33hpZCo670labrD0kBJfqK4bEJHjYFF1ikubFU5A?e=ytsyMx) lists (.txt files) and put them into the `BAM/lists` directory. 

- Run `util/get_mulway_base_data.py` to generate base annotations for pre-training.

### Models

- Download the pre-trained backbones from [here](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/langchunbo_mail_nwpu_edu_cn/EflpnBbWaftEum485cNq8v8BMakzrpvbGfdHWo97FDHYtw?e=m9v2UK) and put them into the `BAM/initmodel` directory. 
- Download our trained base learners from [OneDrive](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/langchunbo_mail_nwpu_edu_cn/ETERT3xe5ndEpDhStts7JmcBFuE3XEqHYKlYdO-Uu96jLg?e=gJLkiT) and put them under `initmodel/PSPNet`. 
- We provide 4 trained BAM [models](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/langchunbo_mail_nwpu_edu_cn/EWjRvUVQYttHkjgg0DyHi4YBPDzt62zix1hPIxdRbuCU7g?e=I2ypgQ) for performance evaluation: 2 VGG16 based models for PASCAL-5<sup>0</sup> and 2 ResNet50 based models for COCO-20<sup>0</sup>.

### Usage

- Change configuration via the `.yaml` files in `BAM/config`, then run the `.sh` scripts for training and testing.

- **Stage1** *Pre-training*

  Train the base learner within the standard learning paradigm.

  ```
  sh train_base.sh
  ```

- **Stage2** *Meta-training*

  Train the meta learner and ensemble module within the meta-learning paradigm. 

  ```
  sh train.sh
  ```

- **Stage3** *Meta-testing*

  Test the proposed model under the standard few-shot setting. 

  ```
  sh test.sh
  ```

- **Stage4** *Generalized testing*

  Test the proposed model under the generalized few-shot setting. 

  ```
  sh test_GFSS.sh
  ```

### Performance

Performance comparison with the state-of-the-art approaches (*i.e.*, [HSNet](https://github.com/juhongm999/hsnet) and [PFENet](https://github.com/dvlab-research/PFENet)) in terms of **average** **mIoU** across all folds. 

1. ##### PASCAL-5<sup>i</sup>

   | Backbone | Method     | 1-shot                   | 5-shot                   |
   | -------- | ---------- | ------------------------ | ------------------------ |
   | VGG16    | HSNet      | 59.70                    | 64.10                    |
   |          | BAM (ours) | 64.41 <sub>(+4.71)</sub> | 68.76 <sub>(+4.66)</sub> |
   | ResNet50 | HSNet      | 64.00                    | 69.50                    |
   |          | BAM (ours) | 67.81 <sub>(+3.81)</sub> | 70.91 <sub>(+1.41)</sub> |

2. ##### COCO-20<sup>i</sup>

   | Backbone | Method     | 1-shot                   | 5-shot                   |
   | -------- | ---------- | ------------------------ | ------------------------ |
   | VGG16    | PFENet     | 36.30                    | 40.40                    |
   |          | BAM (ours) | 43.50 <sub>(+7.20)</sub> | 49.34 <sub>(+8.94)</sub> |
   | ResNet50 | HSNet      | 39.20                    | 46.90                    |
   |          | BAM (ours) | 46.23 <sub>(+7.03)</sub> | 51.16 <sub>(+4.26)</sub> |

### Visualization

<p align="middle">
    <img src="figure/visualization.jpg">
</p>

### References

This repo is mainly built based on [PFENet](https://github.com/dvlab-research/PFENet), [RePRI](https://github.com/mboudiaf/RePRI-for-Few-Shot-Segmentation), and [SemSeg](https://github.com/hszhao/semseg). Thanks for their great work!

### To-Do List

- [x] Support different backbones
- [x] Support various annotations for training/testing
- [x] Multi-GPU training
- [ ] FSS-1000 dataset

### Bibtex

Please consider citing our paper if the project helps your research. BibTeX reference is as follows:

```
@InProceedings{lang2022bam,
    title={Learning What Not to Segment: A New Perspective on Few-Shot Segmentation},
    author={Lang, Chunbo and Cheng, Gong and Tu, Binfei and Han, Junwei},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
}
```

