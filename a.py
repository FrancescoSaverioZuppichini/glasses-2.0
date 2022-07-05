from glasses.models.vision.image.classification.vit.zoo import zoo
from rich.console import Console

zoo["vit_small_patch16_224"]().summary()
