In order to properly create a truly amazing codebase, we must agree on some conding conventions.

We follow the [**PEP 8**](https://peps.python.org/pep-0008/) style guide for Python Code. 

## Naming Convention
### Names

We are lazy programmers! Keep the variables names short but meaningfull:

  - `convolution -> conv`
  - `activation -> act`
  - `linear/dense -> fc`
  - `batchnorm -> bn` 

... etc


### Models

- `Block`: we refer to `Block` to mean a minimum building block of a model.
- `Stage`: a `Stage` is a collection of `Block`s 

A model is always linked to a task. Therefore, each model follows the `<ModelName>For<TaskName>` naming convention. FOr example, `ResNetForImageClassification`.

When you have something very simple, you can directly subclass `nn.Sequential` to avoid writing a trivial `forward` function.

```python
# This module is trivial, use nn.Sequential instead!
class ConvBnReLU(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.conv = nn.Conv2d(...)
        self.bn = nn.BatchNorm2d(...)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# no need for the forward
class ConvBnReLU(nn.Sequential):
    def __init__(self, ...):
        super().__init__()
        self.conv = nn.Conv2d(...)
        self.bn = nn.BatchNorm2d(...)
        self.relu = nn.ReLU()

```

## Design
We follow the single responsability principle where each class/function does only one thing. For example, assume we have the following model (we are subclassing `nn.Sequential` for simplicity).

```python

class MyModel(nn.Module):
    def __init__(...):
        super().__init__()
        self.embedder = nn.Sequential(
            nn.Conv2d(....),
            nn.BatchNorm2d(...),
            nn.ReLU()
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(...),
            nn.Conv2d(...),
            nn.Conv2d(...)
        )
        self.head = nn.Sequential(
            nn.Linear(...)
        )
```

```python

class MyModelEmbedder(nn.Sequential):
    def __init__(...):
        super().__init__(
            nn.Conv2d(...),
            nn.BatchNorm2d(...),
            nn.ReLU()
        )

class MyModelEncoder(nn.Sequential):
    def __init__(...):
        super().__init__(
            nn.Conv2d(...),
            nn.Conv2d(...),
            nn.Conv2d(...)
        )


class MyModelHead(nn.Sequential):
    def __init__(...):
        super().__init__(nn.Linear(...))


class MyModel(nn.Module):
    def __init__(...):
        super().__init__()
        self.embedder = MyModelEmbedder(...)
        self.encoder = MyModelEncoder(...)
        self.head = MyModelHead(...)
```

Each module does only one thing. This makes it easier to document and share each individual parts.