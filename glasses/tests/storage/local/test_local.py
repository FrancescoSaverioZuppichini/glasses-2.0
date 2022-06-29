from pathlib import Path

from glasses.storage import LocalStorage
from glasses.tests.fixtures import TestAutoModel


def test_local(tmp_path: Path, test_auto_model: TestAutoModel):
    storage = LocalStorage(root=tmp_path)
    name = "test1"
    cfg = test_auto_model.get_config_from_name(name)
    model = test_auto_model.from_name(name).eval()
    storage.put(name, model.state_dict(), cfg)

    assert (tmp_path / name).exists()

    state_dict, config = storage.get(name)
    model.load_state_dict(state_dict)
    config.build()
