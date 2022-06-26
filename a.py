import torch

from glasses.models.vision.auto import AutoModelForClassification

model = AutoModelForClassification.from_name("vit_base_patch16_224")
x = torch.randn(2, 3, 224, 224)
d = model(x)
print(model)

print(d)
