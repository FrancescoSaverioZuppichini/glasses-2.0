Glasses uses a configuration system to record/share and load custom version of a specific architecture. 

The main idea behind our configuration system is to be an addition to the models, **not a requirement**. Any model in classes can be created by just importing it and passing the right parameters, they don't know about configurations.

Saying that, why do we need configurations? 

Configurations are necessary when we need to store a specific set of parameters for a model. For example, if a model was trained on dataset `X` with ten classes, our configuration will contain all the parameters need to create that specific model.

In most libraries, configuration are serialized files (e.g. `yml`), in glasses they are piece of code. This allow the user to take advante of it's IDE and see the parameters at any point in time.


Let's see how to create a basic config. First, we need a model

```python
from torch import nn
class MyModel(nn.Module):
    def __init__(in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
```

Then we can create it's configuration

```python
from glasses.config import Config
# Let's create it's configuration
@dataclass
class MyConfig(Config):
    in_channels: int
    out_channels: int

    def build(self) -> nn.Module:
        # create a `MyModel` instance using `MyConfig`
        return MyModel(**self.__dict__)
```

We can now invoke the `build` method, that will create the model

```python
model: MyModel = MyConfig(2, 2).build()
# same as
model: MyModel = MyModel(2, 2)
```

Nothing very special. Let's see how to create a **nested config**. 

Assume we have a model takes a backbone and idk has a fixed head.

```python
from torch import nn

class MyModel(nn.Module):
    def __init__(self, backbone: nn.Module, channels: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features[-1]) # use last feature
        return out
```

Our config will be nested

```python
from glasses.config import Config
@dataclass
class MyConfig(Config):
    backbone_config: Config
    channels: int
    num_classes: int

    def build(self) -> nn.Module:
        backbone = backbone_config.build()
        # create a `MyModel` instance using `MyConfig`
        return MyModel(backbone, self.channels, self.num_classes)
```

Obliously, we must have configs for the different backbones we want to use.


```python
from torch import nn
from glasses.config import Config

class BackboneA(nn.Module):
    def __init__(...):
        ...

@dataclass
class BackboneAConfig(Config):
    ...

class BackboneB(nn.Module):
    def __init__(...):
        ....

@dataclass
class BackboneBConfig(Config):
    ...
```

Then, we can pass any backbone to `MyConfig`.

```python
config = MyConfig(backbone_config=BackboneAConfig(...), ...)
config.build() # build model with backbone A
config = MyConfig(backbone_config=BackboneBConfig(...), ...)
config.build() # build model with backbone B
```
The main advantage of the config system is when we need to save specific model version. For instance, assume I have trained `MyModel` with `BackboneA` on dataset `X`. It's config will look like:


```python
my_model_backbone_a_x = MyConfig(backbone_config=BackboneAConfig(...), channels=64, num_classes=10)

```

Therefore, at any point in time I can recreate the model and load it's pretrained weights.

```python
my_model_backbone_a_x.build().load_state_dict("/somewhere/my_model_backbone_a_x.pth")
```

Now, what if I want to use `my_model_backbone_a_x` architecture but just change a small part? Maybe the numer of classes?


```python
# clone the config
config = MyConfig(**my_model_backbone_a_x.__dict__)
config.num_classes = 8
# load the pretrained weight, with a different number of classes
config.build().load_state_dict("/somewhere/my_model_backbone_a_x.pth")
```

