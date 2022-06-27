from typing import List

from einops.layers.torch import Reduce
from torch import Tensor, nn

from glasses.nn import Lambda

from ..base import HeadForClassification


class ViTHead(HeadForClassification):
    POLICIES = ["token", "mean"]

    def __init__(
        self, emb_size: int = 768, num_classes: int = 1000, policy: str = "token"
    ):
        """
        ViT Classification Head
        Args:
            emb_size (int, optional):  Embedding dimensions Defaults to 768.
            num_classes (int, optional): [description]. Defaults to 1000.
            policy (str, optional): Pooling policy, can be token or mean. Defaults to 'token'.
        """
        if policy not in self.POLICIES:
            raise ValueError(f"Only policies {','.join(self.POLICIES)} are supported")

        super().__init__()
        self.pool = (
            Reduce("b n e -> b e", reduction="mean")
            if policy == "mean"
            else Lambda(lambda x: x[:, 0])
        )
        self.fc = nn.Linear(emb_size, num_classes)

    def forward(self, features: List[Tensor]) -> Tensor:
        x = self.pool(features[-1])
        x = self.fc(x)
        return x
