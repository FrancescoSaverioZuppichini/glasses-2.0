from glasses.models.auto import AutoModel


def test_auto_model(test_config, test_model_func, test_auto_model: AutoModel):

    model = test_auto_model.from_name("test1")

    assert isinstance(model, test_model_func)

    config = test_auto_model.get_config_from_name("test1")

    assert isinstance(config, type(test_config))
    assert config == test_config
