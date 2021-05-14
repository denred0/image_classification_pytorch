# Image classification with pretrained models in Pytorch

Use pretrained models to train your data.

## Installation

```sh
pip install image-classification-pytorch
```

## Get Started

```sh
import image_classification_pytorch as icp

# Images of each class in its own folder with the class name. 
# Examples of folders: homer_simpson, bart_simpson etc. 
from google.colab import drive
drive.mount('/content/gdrive')
!unzip -q /content/gdrive/My\ Drive/datasets/image_classification_pytorch/data_simpsons.zip -d train

# add model
tf_efficientnet_b4_ns = {'model_type': 'tf_efficientnet_b4_ns', 'im_size': 380, 'im_size_test': 380, 'batch_size': 8, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
models = [tf_efficientnet_b4_ns]

# create trainer
trainer = ICPTrainer(models=models, data_dir='data_simpsons')

# start training
trainer.fit_test()
```

