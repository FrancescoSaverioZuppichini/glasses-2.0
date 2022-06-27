from glasses.models.auto.utils import get_names_to_configs_map


def test_get_names_to_configs_map():
    zoo = get_names_to_configs_map("package")
    assert "a1" in zoo
    assert "a2" in zoo
    assert "b1" in zoo
