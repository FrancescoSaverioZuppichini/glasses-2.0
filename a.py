from rich.console import Console

from glasses.models.vision.image.classification.vit.zoo import zoo

zoo["vit_small_patch16_224"]().summary()
