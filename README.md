# EHTDI
Exploring High-quality Target Domain Information for Unsupervised Domain Adaptive Semantic Segmentatio.

In this paper, we construct an efficient non-distillation method for UDA Semantic Segmentation. 

#### Initial Models
 * For GTA5-to-Cityscapes, we start with a model pretrained on the source (GTA5): [Download](https://drive.google.com/file/d/1lpMUoDKZHhoAtx-LRvgkNHdQ7Uq_I7u1/view?usp=sharing)
 * For SYNTHIA-to-Cityscapes, we start with a model pretrained on the source (SYNTHIA): [Download](https://drive.google.com/file/d/1Xuo0WAJosoJP37PAsvaPzczw6v64fVe3/view?usp=sharing)


#### Pretrained models
 * GTA5-to-Cityscapes: [Download](https://drive.google.com/file/d/1vNQHBitIDAiuY8IkmRDfVBShWX6qDiaC/view?usp=sharing)
 * SYNTHIA-to-Cityscapes: [Download](https://drive.google.com/file/d/1ICHI3mDpIQn82o5Q-VFOtPPEMLK-Ijf9/view?usp=sharing)
## Main Results

### GTA5-to-CityScapes and SYNTHIA-to-CityScapes
|                      |   GTA5-to-CityScapes|   |   SYNTHIA-to-CityScapes| |
|----------------------|---------------------|---|---------------------|-|
|                      |mIoU                 ||mIoU_13|mIoU_16|
| Ours        |58.8|   | 57.8 | 64.6  |
| Ours*        |62.0|   | 61.2 | 69.2  |

*Indicates additional edge enhancement loss is added. This part of the code will be released later.

## Acknowledgments

This code is based on the implementations of [**PixMatch: Unsupervised Domain Adaptation via Pixelwise Consistency Training**](https://github.com/lukemelas/pixmatch) and  [**DACS: Domain Adaptation via Cross-domain Mixed Sampling**](https://github.com/vikolss/DACS).
