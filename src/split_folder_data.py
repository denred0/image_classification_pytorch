# import splitfolders
# splitfolders.ratio('data_crop', output="output", seed=1337, ratio=(.8, .2, .0))

from pytorch_lightning.metrics import ConfusionMatrix
import torch

target = torch.tensor([1, 2, 0, 0, 0, 2, 1, 1, 1, 0])
preds = torch.tensor([2, 1, 0, 2, 0, 1, 2, 1, 2, 2])
confmat = ConfusionMatrix(num_classes=3)
t = confmat(preds, target)
print(t.tolist())

print(t)
s = ''
for i in range(t.shape[0]):
    for j in range(t.shape[1]):
        s += str(int(t[i, j].item())) + '_'
