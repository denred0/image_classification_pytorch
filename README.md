# Image Classification PyTorch

Library for quick training models for image classification.


## Table of content
- [Installation](#installation)
- [Quick start](#quick-start)
- [Prediction](#prediction)
- [Parameters](#parameters)
- [Data preparation](#data-preparation)
- [Models](#models)
- [License](#license)

---


## Installation <a name="installation"></a>

```python
pip install image-classification-pytorch
```
---

## Quick Start <a name="quick-start"></a>

```python
import image_classification_pytorch as icp

# add model
# your can add several models for consistent training
tf_efficientnet_b4_ns = {'model_type': 'tf_efficientnet_b4_ns', 'im_size': 380, 'im_size_test': 380, 'batch_size': 8, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
models = [tf_efficientnet_b4_ns]

# create trainer
trainer = ICPTrainer(models=models, data_dir='my_data')
# start training
trainer.fit_test()
```

#### Example 
[Simple example of training and prediction](https://github.com/denred0/image_classification_pytorch/blob/master/examples/image_classification_pytorch_get_started.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1M7oJDizCOrFTDJz0CaDy-ClvDMUvmlnv?usp=sharing)

---

## Prediction <a name="prediction"></a>

Put folders with samples in a folder (data_dir). You can use class labels for folder names.

Example of folders structure

    ├── inference                    # data_dir folder
        ├── dogs                     # Folder Class 1
        ├── cats                     # Folder Class 2


Use the same parameters as for training.
```python
import image_classification_pytorch as icp

icp.ICPInference(data_dir='inference',
                 img_size=380,
                 show_accuracy=True,
                 checkpoint='tb_logs/tf_efficientnet_b4_ns/version_4/checkpoints/tf_efficientnet_b4_ns__epoch=2_val_loss=0.922_val_acc=0.830_val_f1_epoch=0.000.ckpt',
                 std=[0.229, 0.224, 0.225],
                 mean=[0.485, 0.456, 0.406],
                 confidence_threshold=1).predict()
```

After prediction you can see such folders structure

    ├── inference                    # data_dir folder
        ├── dogs                     # Initial dogs folder 
        ├── dogs_gt___dogs           # In this folder should be dogs pictures (ground truth(gt) dogs) and they predicted as dogs
        ├── dogs_gt___cats           # In this folder should be dogs pictures (ground truth(gt) dogs) but they predicted as cats
        ├── cats                     # Initial cats folder
        ├── cats_gt___cats           # In this folder should be cats pictures (ground truth(gt) cats) and they predicted as cats

As you can see all cats predicted as cats and some dogs predicted as cats. 

---

## Data Preparation <a name="data-preparation"></a>
Prepare data for training in the following format

    ├── animals                      # Data folder
        ├── dogs                     # Folder Class 1
        ├── cats                     # Folder Class 2
        ├── gray_bears               # Folder Class 3
        ├── zebras                   # Folder Class 4
        ├── ...
 
---

## Parameters <a name="parameters"></a>

```python

```

---

## Models <a name="models"></a>
323 models out of the box. 
Used models from [PyTorch Image Models (timm)](https://github.com/rwightman/pytorch-image-models)
<details>
<summary style="margin-left: 25px;"><strong>Models</strong></summary>
<ol>
    <li>adv_inception_v3</li>
    <li>cspdarknet53</li>
    <li>cspresnet50</li>
    <li>cspresnext50</li>
    <li>densenet121</li>
    <li>densenet161</li>
    <li>densenet169</li>
    <li>densenet201</li>
    <li>densenetblur121d</li>
    <li>dla102</li>
    <li>dla102x</li>
    <li>dla102x2</li>
    <li>dla169</li>
    <li>dla34</li>
    <li>dla46_c</li>
    <li>dla46x_c</li>
    <li>dla60_res2net</li>
    <li>dla60_res2next</li>
    <li>dla60</li>
    <li>dla60x_c</li>
    <li>dla60x</li>
    <li>dm_nfnet_f0</li>
    <li>dm_nfnet_f1</li>
    <li>dm_nfnet_f2</li>
    <li>dm_nfnet_f3</li>
    <li>dm_nfnet_f4</li>
    <li>dm_nfnet_f5</li>
    <li>dm_nfnet_f6</li>
    <li>dpn107</li>
    <li>dpn131</li>
    <li>dpn68</li>
    <li>dpn68b</li>
    <li>dpn92</li>
    <li>dpn98</li>
    <li>ecaresnet101d_pruned</li>
    <li>ecaresnet101d</li>
    <li>ecaresnet269d</li>
    <li>ecaresnet26t</li>
    <li>ecaresnet50d_pruned</li>
    <li>ecaresnet50d</li>
    <li>ecaresnet50t</li>
    <li>ecaresnetlight</li>
    <li>efficientnet_b0</li>
    <li>efficientnet_b1_pruned</li>
    <li>efficientnet_b1</li>
    <li>efficientnet_b2</li>
    <li>efficientnet_b2a</li>
    <li>efficientnet_b3_pruned</li>
    <li>efficientnet_b3</li>
    <li>efficientnet_b3a</li>
    <li>efficientnet_em</li>
    <li>efficientnet_es</li>
    <li>efficientnet_lite0</li>
    <li>ens_adv_inception_resnet_v2</li>
    <li>ese_vovnet19b_dw</li>
    <li>ese_vovnet39b</li>
    <li>fbnetc_100</li>
    <li>gernet_l</li>
    <li>gernet_m</li>
    <li>gernet_s</li>
    <li>gluon_inception_v3</li>
    <li>gluon_resnet101_v1b</li>
    <li>gluon_resnet101_v1c</li>
    <li>gluon_resnet101_v1d</li>
    <li>gluon_resnet101_v1s</li>
    <li>gluon_resnet152_v1b</li>
    <li>gluon_resnet152_v1c</li>
    <li>gluon_resnet152_v1d</li>
    <li>gluon_resnet152_v1s</li>
    <li>gluon_resnet18_v1b</li>
    <li>gluon_resnet34_v1b</li>
    <li>gluon_resnet50_v1b</li>
    <li>gluon_resnet50_v1c</li>
    <li>gluon_resnet50_v1d</li>
    <li>gluon_resnet50_v1s</li>
    <li>gluon_resnext101_32x4d</li>
    <li>gluon_resnext101_64x4d</li>
    <li>gluon_resnext50_32x4d</li>
    <li>gluon_senet154</li>
    <li>gluon_seresnext101_32x4d</li>
    <li>gluon_seresnext101_64x4d</li>
    <li>gluon_seresnext50_32x4d</li>
    <li>gluon_xception65</li>
    <li>hrnet_w18_small_v2</li>
    <li>hrnet_w18_small</li>
    <li>hrnet_w18</li>
    <li>hrnet_w30</li>
    <li>hrnet_w32</li>
    <li>hrnet_w40</li>
    <li>hrnet_w44</li>
    <li>hrnet_w48</li>
    <li>hrnet_w64</li>
    <li>ig_resnext101_32x16d</li>
    <li>ig_resnext101_32x32d</li>
    <li>ig_resnext101_32x48d</li>
    <li>ig_resnext101_32x8d</li>
    <li>inception_resnet_v2</li>
    <li>inception_v3</li>
    <li>inception_v4</li>
    <li>legacy_senet154</li>
    <li>legacy_seresnet101</li>
    <li>legacy_seresnet152</li>
    <li>legacy_seresnet18</li>
    <li>legacy_seresnet34</li>
    <li>legacy_seresnet50</li>
    <li>legacy_seresnext101_32x4d</li>
    <li>legacy_seresnext26_32x4d</li>
    <li>legacy_seresnext50_32x4d</li>
    <li>mixnet_l</li>
    <li>mixnet_m</li>
    <li>mixnet_s</li>
    <li>mixnet_xl</li>
    <li>mnasnet_100</li>
    <li>mobilenetv2_100</li>
    <li>mobilenetv2_110d</li>
    <li>mobilenetv2_120d</li>
    <li>mobilenetv2_140</li>
    <li>mobilenetv3_large_100</li>
    <li>mobilenetv3_rw</li>
    <li>nasnetalarge</li>
    <li>nf_regnet_b1</li>
    <li>nf_resnet50</li>
    <li>nfnet_l0c</li>
    <li>pnasnet5large</li>
    <li>regnetx_002</li>
    <li>regnetx_004</li>
    <li>regnetx_006</li>
    <li>regnetx_008</li>
    <li>regnetx_016</li>
    <li>regnetx_032</li>
    <li>regnetx_040</li>
    <li>regnetx_064</li>
    <li>regnetx_080</li>
    <li>regnetx_120</li>
    <li>regnetx_160</li>
    <li>regnetx_320</li>
    <li>regnety_002</li>
    <li>regnety_004</li>
    <li>regnety_006</li>
    <li>regnety_008</li>
    <li>regnety_016</li>
    <li>regnety_032</li>
    <li>regnety_040</li>
    <li>regnety_064</li>
    <li>regnety_080</li>
    <li>regnety_120</li>
    <li>regnety_160</li>
    <li>regnety_320</li>
    <li>repvgg_a2</li>
    <li>repvgg_b0</li>
    <li>repvgg_b1</li>
    <li>repvgg_b1g4</li>
    <li>repvgg_b2</li>
    <li>repvgg_b2g4</li>
    <li>repvgg_b3</li>
    <li>repvgg_b3g4</li>
    <li>res2net101_26w_4s</li>
    <li>res2net50_14w_8s</li>
    <li>res2net50_26w_4s</li>
    <li>res2net50_26w_6s</li>
    <li>res2net50_26w_8s</li>
    <li>res2net50_48w_2s</li>
    <li>res2next50</li>
    <li>resnest101e</li>
    <li>resnest14d</li>
    <li>resnest200e</li>
    <li>resnest269e</li>
    <li>resnest26d</li>
    <li>resnest50d_1s4x24d</li>
    <li>resnest50d_4s2x40d</li>
    <li>resnest50d</li>
    <li>resnet101d</li>
    <li>resnet152d</li>
    <li>resnet18</li>
    <li>resnet18d</li>
    <li>resnet200d</li>
    <li>resnet26</li>
    <li>resnet26d</li>
    <li>resnet34</li>
    <li>resnet34d</li>
    <li>resnet50</li>
    <li>resnet50d</li>
    <li>resnetblur50</li>
    <li>resnetv2_101x1_bitm_in21k</li>
    <li>resnetv2_101x1_bitm</li>
    <li>resnetv2_101x3_bitm_in21k</li>
    <li>resnetv2_101x3_bitm</li>
    <li>resnetv2_152x2_bitm_in21k</li>
    <li>resnetv2_152x2_bitm</li>
    <li>resnetv2_152x4_bitm_in21k</li>
    <li>resnetv2_152x4_bitm</li>
    <li>resnetv2_50x1_bitm_in21k</li>
    <li>resnetv2_50x1_bitm</li>
    <li>resnetv2_50x3_bitm_in21k</li>
    <li>resnetv2_50x3_bitm</li>
    <li>resnext101_32x8d</li>
    <li>resnext50_32x4d</li>
    <li>resnext50d_32x4d</li>
    <li>rexnet_100</li>
    <li>rexnet_130</li>
    <li>rexnet_150</li>
    <li>rexnet_200</li>
    <li>selecsls42b</li>
    <li>selecsls60</li>
    <li>selecsls60b</li>
    <li>semnasnet_100</li>
    <li>seresnet152d</li>
    <li>seresnet50</li>
    <li>seresnext26d_32x4d</li>
    <li>seresnext26t_32x4d</li>
    <li>seresnext50_32x4d</li>
    <li>skresnet18</li>
    <li>skresnet34</li>
    <li>skresnext50_32x4d</li>
    <li>spnasnet_100</li>
    <li>ssl_resnet18</li>
    <li>ssl_resnet50</li>
    <li>ssl_resnext101_32x16d</li>
    <li>ssl_resnext101_32x4d</li>
    <li>ssl_resnext101_32x8d</li>
    <li>ssl_resnext50_32x4d</li>
    <li>swsl_resnet18</li>
    <li>swsl_resnet50</li>
    <li>swsl_resnext101_32x16d</li>
    <li>swsl_resnext101_32x4d</li>
    <li>swsl_resnext101_32x8d</li>
    <li>swsl_resnext50_32x4d</li>
    <li>tf_efficientnet_b0_ap</li>
    <li>tf_efficientnet_b0_ns</li>
    <li>tf_efficientnet_b0</li>
    <li>tf_efficientnet_b1_ap</li>
    <li>tf_efficientnet_b1_ns</li>
    <li>tf_efficientnet_b1</li>
    <li>tf_efficientnet_b2_ap</li>
    <li>tf_efficientnet_b2_ns</li>
    <li>tf_efficientnet_b2</li>
    <li>tf_efficientnet_b3_ap</li>
    <li>tf_efficientnet_b3_ns</li>
    <li>tf_efficientnet_b3</li>
    <li>tf_efficientnet_b4_ap</li>
    <li>tf_efficientnet_b4_ns</li>
    <li>tf_efficientnet_b4</li>
    <li>tf_efficientnet_b5_ap</li>
    <li>tf_efficientnet_b5_ns</li>
    <li>tf_efficientnet_b5</li>
    <li>tf_efficientnet_b6_ap</li>
    <li>tf_efficientnet_b6_ns</li>
    <li>tf_efficientnet_b6</li>
    <li>tf_efficientnet_b7_ap</li>
    <li>tf_efficientnet_b7_ns</li>
    <li>tf_efficientnet_b7</li>
    <li>tf_efficientnet_b8_ap</li>
    <li>tf_efficientnet_b8</li>
    <li>tf_efficientnet_cc_b0_4e</li>
    <li>tf_efficientnet_cc_b0_8e</li>
    <li>tf_efficientnet_cc_b1_8e</li>
    <li>tf_efficientnet_el</li>
    <li>tf_efficientnet_em</li>
    <li>tf_efficientnet_es</li>
    <li>tf_efficientnet_l2_ns_475</li>
    <li>tf_efficientnet_l2_ns</li>
    <li>tf_efficientnet_lite0</li>
    <li>tf_efficientnet_lite1</li>
    <li>tf_efficientnet_lite2</li>
    <li>tf_efficientnet_lite3</li>
    <li>tf_efficientnet_lite4</li>
    <li>tf_inception_v3</li>
    <li>tf_mixnet_l</li>
    <li>tf_mixnet_m</li>
    <li>tf_mixnet_s</li>
    <li>tf_mobilenetv3_large_075</li>
    <li>tf_mobilenetv3_large_100</li>
    <li>tf_mobilenetv3_large_minimal_100</li>
    <li>tf_mobilenetv3_small_075</li>
    <li>tf_mobilenetv3_small_100</li>
    <li>tf_mobilenetv3_small_minimal_100</li>
    <li>tresnet_l_448</li>
    <li>tresnet_l</li>
    <li>tresnet_m_448</li>
    <li>tresnet_m</li>
    <li>tresnet_xl_448</li>
    <li>tresnet_xl</li>
    <li>tv_densenet121</li>
    <li>tv_resnet101</li>
    <li>tv_resnet152</li>
    <li>tv_resnet34</li>
    <li>tv_resnet50</li>
    <li>tv_resnext50_32x4d</li>
    <li>vgg11_bn</li>
    <li>vgg11</li>
    <li>vgg13_bn</li>
    <li>vgg13</li>
    <li>vgg16_bn</li>
    <li>vgg16</li>
    <li>vgg19_bn</li>
    <li>vgg19</li>
    <li>vit_base_patch16_224_in21k</li>
    <li>vit_base_patch16_224</li>
    <li>vit_base_patch16_384</li>
    <li>vit_base_patch32_224_in21k</li>
    <li>vit_base_patch32_384</li>
    <li>vit_base_resnet50_224_in21k</li>
    <li>vit_base_resnet50_384</li>
    <li>vit_deit_base_distilled_patch16_224</li>
    <li>vit_deit_base_distilled_patch16_384</li>
    <li>vit_deit_base_patch16_224</li>
    <li>vit_deit_base_patch16_384</li>
    <li>vit_deit_small_distilled_patch16_224</li>
    <li>vit_deit_small_patch16_224</li>
    <li>vit_deit_tiny_distilled_patch16_224</li>
    <li>vit_deit_tiny_patch16_224</li>
    <li>vit_large_patch16_224_in21k</li>
    <li>vit_large_patch16_224</li>
    <li>vit_large_patch16_384</li>
    <li>vit_large_patch32_224_in21k</li>
    <li>vit_large_patch32_384</li>
    <li>vit_small_patch16_224</li>
    <li>wide_resnet101_2</li>
    <li>wide_resnet50_2</li>
    <li>xception</li>
    <li>xception41</li>
    <li>xception65</li>
    <li>xception71</li>
  </ol>
</div>
</details>


---

**Get List Models**
```python
import timm
from pprint import pprint
model_names = timm.list_models(pretrained=True)
pprint(model_names)
>>> ['adv_inception_v3',
 'cspdarknet53',
 'cspresnext50',
 'densenet121',
 'densenet161',
 'densenet169',
 'densenet201',
 'densenetblur121d',
 'dla34',
 'dla46_c',
...
]
```

**Get Model parameters**
```python
import timm
from pprint import pprint
m = timm.create_model('efficientnet_b0', pretrained=True)
pprint(m.default_cfg)
```

Timm documentation [here](https://rwightman.github.io/pytorch-image-models/)

---


## License <a name="license"></a>
Project is distributed under [MIT License](https://github.com/denred0/image_classification_pytorch/blob/master/LICENSE.txt)
