from glasses.storage.huggingface import HuggingFaceStorage

# create_repo(
#     repo_id="glasses/hey", token="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", exist_ok=True
# )

from glasses.models.vision.auto import AutoModelBackbone
from glasses.storage.local import LocalStorage

key = "vit_small_patch16_224"
config = AutoModelBackbone.get_config_from_name(key)

model = AutoModelBackbone.from_name(key)

key = "asd"
# storage = LocalStorage(override=True)
storage = HuggingFaceStorage()
print(storage.models)
# storage.put(key, model.state_dict(), config)
# storage.get(key)
# HuggingFaceStorage()
