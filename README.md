# Image Classification PyTorch

Library for quick training models for image classification.


### Table of content
- [Installation](#installation)
- [Quick start](#quick-start)
- [Prediction](#prediction)
- [Parameters](#parameters)
- [Data preparation](#data-preparation)
- [Models](#models)
- [License](#license)

---


### Installation <a name="installation"></a>

```python
pip install image-classification-pytorch
```
---

### Quick Start <a name="quick-start"></a>

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

### Prediction <a name="prediction"></a>

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

### Data Preparation <a name="data-preparation"></a>
Prepare data for training in the following format

    ├── animals                      # Data folder
        ├── dogs                     # Folder Class 1
        ├── cats                     # Folder Class 2
        ├── gray_bears               # Folder Class 3
        ├── zebras                   # Folder Class 4
        ├── ...
 
---

### Parameters <a name="parameters"></a>

```python

```

---

### Models <a name="models"></a>
Used models from [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)

**List Models**
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

**Model parameters**
```python
import timm
from pprint import pprint
m = timm.create_model('efficientnet_b0', pretrained=True)
pprint(m.default_cfg)
```

Timm documentation [here](https://rwightman.github.io/pytorch-image-models/)

---


### License <a name="license"></a>
Project is distributed under [MIT License](https://github.com/denred0/image_classification_pytorch/blob/master/LICENSE.txt)
