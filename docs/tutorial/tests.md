Testing your code is crucial to ensure its correctness. We try to help you smooth this process by proving an easy-to-use recipe. 

Let's see how to add a test for a model to glasses.

## Setup

Let's take an example, you have created a new image classification model called `BoringModel`. Following our structure, its config should be in `glasses/models/vision/image/classification/boring_model/config.py`

```
├── glasses
│   ├── models
│   │   └── vision
│   │       ├── image
│   │       │   ├── classification
│   │       │   │   │   └── boring_model
│   │       │   │   │       ├── config.py
│   │       │   │   │       ├── model.py

```

Its config may look like

```python
# config.py
from dataclasses import dataclass

from glasses.config import Config

from .model import BoringModelForImageClassification


@dataclass
class BoringModelForImageClassificationConfig(Config):
    in_channels: int = 3
    hidden_size: int = 32
    num_classes: int = 10

    def build(self) -> BoringModelForImageClassification:
        return BoringModelForImageClassification(**self.__dict__)
```

and the model

```python
# model.py
from torch import Tensor, nn

from ..base import (ModelForImageClassification,
                    ModelForImageClassificationOutput)


class BoringModelForImageClassification(ModelForImageClassification):
    def __init__(self, in_channels: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden_size, kernel_size=3)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values: Tensor) -> ModelForImageClassificationOutput:
        x = self.conv(pixel_values)
        x = self.avg(x).flatten(1)
        logits = self.fc(x)
        return ModelForImageClassificationOutput(logits=logits)

```

Cool, let's see how to test our `BoringModelForImageClassification`

## Test it!

To mimic our internal structure, we will add a test at `glasses/models/vision/image/classification/boring_model/test_boring_model.py`. If you have used our [`cli`]() you should have it by default.

Since we know which config the model needs and we fixed the output, test it it's straightforward.

We provide you a handy function, [`model_tester`](/reference/utils/model_tester/#model_tester). You need to pass to it your config, the input dictionary, the model type and expected outputs shape and optionally expected outputs values.

So, first create a new config for testing the model

```python
# test_boring_model.py
import torch

from glasses.models.vision.image.classification.boring_model import BoringModelForImageClassificationConfig, BoringModelForImageClassification
from glasses.models.vision.image.classification import \
    ModelForImageClassificationOutput
from tests.model_tester import model_tester


def test_boring_model():
    batch_size = 2
    # create a config for a small model, we need our test to be fast
    config = BoringModelForImageClassificationConfig(
        in_channels=3,
        hidden_size=16,
        num_classes=10
    )
```

Then, create an `input_dict`, this will be passed to your model. *8It must match the `forward` arguments of your model`

Then, the shape of the expected output:

```python
# test_boring_model.py
def test_boring_model():
    #...    
    output_shape_dict = {"logits": (batch_size, config.num_classes)}
```

Finally, we can call [`model_tester`](/reference/utils/model_tester/#model_tester)


```python
# test_boring_model.py
def test_boring_model():
    #...    
    model_tester(
        config, input_dict, output_shape_dict, ModelForImageClassificationOutput
    )
```

All together:

```python
# test_boring_model.py
import torch

from glasses.models.vision.image.classification.boring_model import BoringModelForImageClassificationConfig, BoringModelForImageClassification
from glasses.models.vision.image.classification import \
    ModelForImageClassificationOutput
from tests.model_tester import model_tester


def test_boring_model():
    batch_size = 2
    # create a config for a small model, we need our test to be fast
    config = BoringModelForImageClassificationConfig(
        in_channels=3,
        hidden_size=16,
        num_classes=10
    )
    # create the input dict, something like 
    input_dict = {
        "pixel_values": torch.randn(
            (
                batch_size,
                config.in_channels,
                56,
                56
            )
        )
    }
    output_shape_dict = {"logits": (batch_size, config.num_classes)}
    model_tester(
        config, input_dict, output_shape_dict, ModelForImageClassificationOutput
    )
```

We can now run the tests with `pytest`

<div class="termy">

```console
$ python -m pytest ./tests/models/vision/image/classification/boring_model
=========================================================================================================== test session starts ===========================================================================================================
platform linux -- Python 3.9.12, pytest-7.1.3, pluggy-1.0.0
rootdir: /home/zuppif/Documents/glasses-2.0
collected 1 item                                                                                                                                                                                                                          

tests/models/vision/image/classification/boring_model/test_boring_model.py .                                                                                                                                                        [100%]

============================================================================================================ 1 passed in 0.05s ============================================================================================================
```

</div>

Done it! 🧪🧪🧪