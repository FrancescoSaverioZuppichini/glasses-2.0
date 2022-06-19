from .Config import Config


class Configurable:
    @classmethod
    def from_config(cls, config: Config):
        return cls(**config.__dict__)
