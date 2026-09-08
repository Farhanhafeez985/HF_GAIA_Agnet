import os

from datasets import load_dataset
from huggingface_hub import snapshot_download, hf_hub_download


class GaiaFileManager:
    def __init__(self):
        # Download/cache GAIA dataset
        self.dataset_dir = snapshot_download(
            repo_id="gaia-benchmark/GAIA",
            repo_type="dataset",
        )

        # Load validation dataset
        self.dataset = load_dataset(
            self.dataset_dir,
            "2023_level1",
            split="validation",
        )

        # Create task_id -> file_path mapping
        self.file_map = {
            item["task_id"]: item["file_path"]
            for item in self.dataset
            if item["file_path"]
        }

    def get_file(self, task_id: str) -> str | None:
        """
        Get the local path of the attachment for a task.

        Returns:
            str: Local file path
            None: If task has no attachment
        """

        file_path = self.file_map.get(task_id)

        if not file_path:
            return None

        # Download/cache the file
        local_path = hf_hub_download(
            repo_id="gaia-benchmark/GAIA",
            repo_type="dataset",
            filename=file_path,
        )

        return local_path

    def get_content(self, task_id: str) -> bytes | None:
        """
        Get attachment content as bytes.
        """

        local_path = self.get_file(task_id)

        if not local_path:
            return None

        with open(local_path, "rb") as f:
            return f.read()


# Create once and reuse
gaia_files = GaiaFileManager()


def get_file(task_id: str) -> str | None:
    """
    Convenience function.

    Example:
        file_path = get_file(task_id)
    """
    return gaia_files.get_file(task_id)


def get_content(task_id: str) -> bytes | None:
    """
    Convenience function.

    Example:
        content = get_content(task_id)
    """
    return gaia_files.get_content(task_id)


# ================== GET LOCAL FILE PATH  ================== #
# from gaia_files import get_file
task_id = "8e867cd7-cff9-4e6c-867a-ff5ddc2550be"
file_path = get_file(task_id)
print(file_path)

# ==================== For raw content ==================== #

# from gaia_files import get_content
# content = get_content(task_id)
# if content:
#     print(len(content))
