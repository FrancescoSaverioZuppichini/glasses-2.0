## Naming Convention
### Names
Keep the variables names short but meaningfull 
  - `convolution -> conv`
  - `activation -> act`
  - `linear/dense -> fc`
  - `batchnorm -> bn` .. etc

### Design
We follow the single responsability principle where each class/function does only one thing. Here we are subclassing `nn.Sequential` for simplicity.

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


