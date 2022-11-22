# EHTDI
Official implementation of "Exploring High-quality Target Domain Information for Unsupervised Domain Adaptive Semantic Segmentation". [here](https://arxiv.org/abs/2208.06100)

Accepted by ACM MM'22.

#### Our method does not require the distillation technique and requires only 1*3090 GPU.
#### News!!!!!!!!!!!!
We released the code for edge_enhencemant loss

## Main Results

### GTA5-to-CityScapes and SYNTHIA-to-CityScapes
|                      |   GTA5-to-CityScapes|   |   SYNTHIA-to-CityScapes| |
|----------------------|---------------------|---|------------------------|-|
|                      |mIoU                 |   |mIoU_13  (mIoU_16)|
| Ours        |58.8 |   [Model](https://drive.google.com/file/d/1vNQHBitIDAiuY8IkmRDfVBShWX6qDiaC/view?usp=sharing)|  64.6 (57.8) |[Model](https://drive.google.com/file/d/1ICHI3mDpIQn82o5Q-VFOtPPEMLK-Ijf9/view?usp=sharing)  |
| Ours*        |62.0|  [Model](https://drive.google.com/file/d/1YmgnjG2bBIP7U1Egj2Yka4NCXGcF0ctd/view?usp=sharing) |  69.2    (61.3)  |[Model](https://drive.google.com/file/d/1MLh61JU8JGfgdeBWnylFXMjhMh49lasa/view?usp=sharing)  |

*Indicates a new edge enhancement loss is added and still no distillation technology is required. 

#### Data Preparation
To run on GTA5-to-Cityscapes and SYNTHIA-to-Cityscapes, you need to download the respective datasets. Once they are downloaded, you can either modify the config files directly, or organize/symlink the data in the `datasets/` directory as follows: 
```
datasets
├── cityscapes
│   ├── gtFine
│   │   ├── train
│   │   │   ├── aachen
│   │   │   └── ...
│   │   └── val
│   └── leftImg8bit
│       ├── train
│       └── val
├── GTA5
│   ├── images
│   ├── labels
│   └── list
├── SYNTHIA
│   └── RAND_CITYSCAPES
│       ├── Depth
│       │   └── Depth
│       ├── GT
│       │   ├── COLOR
│       │   └── LABELS
│       ├── RGB
│       └── synthia_mapped_to_cityscapes
├── city_list
├── gta5_list
└── synthia_list
```

### create symbolic link：
```
ln -s /data/lijj/pixmatch_output_61.5/ ./outputs
```


#### environment
```
requirement.txt
```

#### Initial Models
* ImageNet pretrain: [Download](http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth)
 * For GTA5-to-Cityscapes, we start with a model pretrained on the source (GTA5): [Download](https://drive.google.com/file/d/1lpMUoDKZHhoAtx-LRvgkNHdQ7Uq_I7u1/view?usp=sharing)
 * For SYNTHIA-to-Cityscapes, we start with a model pretrained on the source (SYNTHIA): [Download](https://drive.google.com/file/d/1Xuo0WAJosoJP37PAsvaPzczw6v64fVe3/view?usp=sharing)


### training
```
sh train.sh
```

## Acknowledgments

This code is based on the implementations of [**PixMatch: Unsupervised Domain Adaptation via Pixelwise Consistency Training**](https://github.com/lukemelas/pixmatch) and  [**DACS: Domain Adaptation via Cross-domain Mixed Sampling**](https://github.com/vikolss/DACS).


