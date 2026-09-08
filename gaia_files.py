from pathlib import Path
from datasets import load_dataset
from huggingface_hub import snapshot_download


class GaiaFileManager:
    def __init__(self):
        self.dataset_dir = snapshot_download(
            repo_id="gaia-benchmark/GAIA",
            repo_type="dataset",
        )

        self.dataset = load_dataset(
            self.dataset_dir,
            "2023_level1",
            split="validation",
        )

        self.file_map = {
            item["task_id"]: item["file_path"]
            for item in self.dataset
            if item["file_path"]
        }

    def get_file(self, task_id: str) -> str | None:
        file_path = self.file_map.get(task_id)

        if not file_path:
            return None

        local_path = Path(self.dataset_dir) / file_path

        if not local_path.exists():
            return None

        return str(local_path)


gaia_files = GaiaFileManager()


def get_file(task_id: str) -> str | None:
    return gaia_files.get_file(task_id)