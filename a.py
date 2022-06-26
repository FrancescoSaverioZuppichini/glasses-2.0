import torch

from glasses.models.vision.auto import AutoModelForClassification

test_set = [
    "levit_128S",
    "levit_128",
    "levit_192",
    "levit_256",
    "levit_384",
]

x = torch.randn(2, 3, 224, 224)

for each in test_set:
    model = AutoModelForClassification.from_name(each)
    print(each, model)
    outputs = model(x)
    print(each, outputs['logits'].shape)
