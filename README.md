# EHTDI
Official implementation of "Exploring High-quality Target Domain Information for Unsupervised Domain Adaptive Semantic Segmentation". [here](https://arxiv.org/abs/2208.06100)

Accepted by ACM MM'22.

#### Our method does not require the distillation technique and requires only 1*3090 GPU.

## Main Results

### GTA5-to-CityScapes and SYNTHIA-to-CityScapes
|                      |   GTA5-to-CityScapes|   |   SYNTHIA-to-CityScapes| |
|----------------------|---------------------|---|------------------------|-|
|                      |mIoU                 |   |mIoU_13  (mIoU_16)|
| Ours        |58.8 |   [Model](https://drive.google.com/file/d/1vNQHBitIDAiuY8IkmRDfVBShWX6qDiaC/view?usp=sharing)|  64.6 (57.8) |[Model](https://drive.google.com/file/d/1ICHI3mDpIQn82o5Q-VFOtPPEMLK-Ijf9/view?usp=sharing)  |
| Ours*        |62.0|  [Model](https://drive.google.com/file/d/1YmgnjG2bBIP7U1Egj2Yka4NCXGcF0ctd/view?usp=sharing) |  69.2    (61.3)  |[Model](https://drive.google.com/file/d/1MLh61JU8JGfgdeBWnylFXMjhMh49lasa/view?usp=sharing)  |

*Indicates additional edge enhancement loss is added. This part of the training code will be released later.

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
pytorch1.10.1+cu113
absl-py                 1.0.0
albumentations          1.1.0
antlr4-python3-runtime  4.8
cachetools              4.2.4
certifi                 2021.10.8
charset-normalizer      2.0.10
google-auth             2.3.3
google-auth-oauthlib    0.4.6
grpcio                  1.43.0
hydra-core              1.1.1
idna                    3.3
imageio                 2.14.0
importlib-metadata      4.10.1
importlib-resources     5.4.0
joblib                  1.1.0
Markdown                3.3.6
networkx                2.6.3
numpy                   1.21.5
oauthlib                3.1.1
omegaconf               2.1.1
opencv-python           4.5.5.62
opencv-python-headless  4.5.5.62
packaging               21.3
Pillow                  9.0.0
pip                     21.2.2
protobuf                3.19.3
pyasn1                  0.4.8
pyasn1-modules          0.2.8
pyparsing               3.0.7
PyWavelets              1.2.0
PyYAML                  6.0
qudida                  0.0.4
requests                2.27.1
requests-oauthlib       1.3.0
rsa                     4.8
scikit-image            0.19.1
scikit-learn            1.0.2
scipy                   1.7.3
setuptools              58.0.4
six                     1.16.0
tensorboard             2.8.0
tensorboard-data-server 0.6.1
tensorboard-plugin-wit  1.8.1
tensorboardX            2.4.1
threadpoolctl           3.0.0
tifffile                2021.11.2
torch                   1.10.1+cu113
torchaudio              0.10.1+cu113
torchvision             0.11.2+cu113
tqdm                    4.62.3
typing_extensions       4.0.1
urllib3                 1.26.8
Werkzeug                2.0.2
wheel                   0.37.1
zipp                    3.7.0

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


