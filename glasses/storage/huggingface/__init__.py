from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from typing import Dict, List, Tuple

import requests
from huggingface_hub.hf_api import ENDPOINT, HfFolder, create_repo, upload_folder
from huggingface_hub.repository import Repository
from requests import Response

from glasses.config import Config
from glasses.types import StateDict

from ..base import Storage
from ..local import LocalStorage


@dataclass
class HuggingFaceStorage(LocalStorage):
    organization: str = "glasses"

    def __post_init__(self):
        self._hf_folder = HfFolder()
        self._local_storage = LocalStorage(self.root, self.override, self.fmt)

    def put(self, key: str, state_dict: StateDict, config: Config):
        repo_id = f"{self.organization}/{key}"
        token = self._hf_folder.get_token()
        create_repo(repo_id, token=token, exist_ok=True)
        # we use LocalStorage apis to create the folder
        self._local_storage.put(key, state_dict, config)
        folder_path = f"{self._local_storage.root}/{key}"
        # we upload that folder to hugging face hub
        upload_folder(repo_id=repo_id, folder_path=folder_path, path_in_repo=".")

        rmtree(folder_path)

    def get(self, key: str) -> Tuple[StateDict, Config]:
        cached_dir = f"{self._local_storage.root}/{key}"
        # create if it doesn't exist
        Path(cached_dir).mkdir(parents=True, exist_ok=True)
        clone_from = f"https://huggingface.co/{self.organization}/{key}"
        # using hf shitty api to instantiate a repo and clone it
        Repository(cached_dir, clone_from=clone_from)
        # we use LocalStorage to read the content of that folder and return what we need
        return self._local_storage.get(key)

    @property
    def models(self) -> List[str]:
        res: Response = requests.get(
            f"{ENDPOINT}/api/models", params={"author": self.organization}
        )
        res.raise_for_status()
        models: List[Dict] = res.json()

        # modelId has the following form: <ORGANIZATION>/<REPO_NAME>/<FILE_NAME>
        names: List[str] = [e["modelId"].split("/")[1] for e in models]

        return names

    def __contains__(self, key: str) -> bool:
        return key in self.models
