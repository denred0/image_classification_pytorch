import timm
from pprint import pprint

model_names = timm.list_models(pretrained=True)

# pprint(model_names)

model = 'vgg16'
m = timm.create_model(model, pretrained=True)
print(model)
pprint(m.default_cfg)



