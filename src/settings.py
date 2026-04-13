from pathlib import Path
import json

REQUIRED_METADATA_FIELDS = [
    "basic_info.id",
    "basic_info.name",
    "basic_info.short_name",
    "basic_info.description",
    "project_context.task",
    "statistics.n_images",
    "statistics.n_annotations",
    "license",
    "links.github",
    "links.download",
    "citation.bibtex",
]


def _get_nested(obj, dotted):
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def load_metadata(repo_root: Path):
    metadata_path = repo_root / "METADATA.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_metadata(repo_root: Path):
    metadata = load_metadata(repo_root)
    missing = []
    for field in REQUIRED_METADATA_FIELDS:
        val = _get_nested(metadata, field)
        if val in (None, "", []):
            missing.append(field)
    return missing
