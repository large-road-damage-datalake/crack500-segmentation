from pathlib import Path


def convert_and_prepare_dataset(repo_root: Path):
    """
    Placeholder for dataset-specific conversion logic.

    Each dataset repo can implement source-to-standard conversion here
    before running the stats/visualization build pipeline.
    """
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
