import torch

from glasses.models.vision.auto import AutoModelForImageClassification
from glasses.storage.local import LocalStorage

"resnet50-base-patch16-in22k"
"resnet50-in1k"
{"resnet50": ["in1k", "in22k"]}
# get the model name -> build
model = AutoModelForImageClassification.from_pretrained("resnet50-custom", config, storage=local)
# <model_name> /<weight_name>.pt
#              /config.pt
#  resnet_base_something-in22k
# resnet50-in22k /
#                 model.pt
#                 config.pkl

x = torch.randn(2, 3, 224, 224)
d = model(x)
# in the zoo we don't have a config for "resnet50-in22k"

AutoModelForImageClassification.from_pretrained(
    "resnet50:in22k",
)

for name, param in model.named_parameters():
    print(name, param.grad)

print(d["logits"].shape)
print((d["logits"]).mean().backward())

for name, param in model.named_parameters():
    print(name, param.grad)
# print(d["logits"].backward(torch.zeros((2, 1000))))
