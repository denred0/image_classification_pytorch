# Image classification with pretrained models in Pytorch

Use pretrained models to train your data.

## Installation

```sh
pip install image-classification-pytorch
```

## Example

```sh
tf_efficientnet_b4_ns = {'model_type': 'tf_efficientnet_b4_ns', 'im_size': 380, 'im_size_test': 380, 'batch_size': 8, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
models = [tf_efficientnet_b4_ns]
trainer = ICPTrainer(models=models, data_dir='data_simpsons')
trainer.fit_test()
```