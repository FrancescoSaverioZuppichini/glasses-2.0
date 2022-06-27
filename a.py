import torch

from glasses.models.vision.auto import AutoModelForClassification

model = AutoModelForClassification.from_name("vit_base_patch16_224")
x = torch.randn(2, 3, 224, 224)
d = model(x)

for name, param in model.named_parameters():
    print(name, param.grad)

print(d["logits"].shape)
print((d["logits"]).mean().backward())

for name, param in model.named_parameters():
    print(name, param.grad)
# print(d["logits"].backward(torch.zeros((2, 1000))))
