import argparse
from pathlib import Path

from src.convert import convert_and_prepare_dataset
from src.options import DOC_FILES, STATS_FILES, VIS_FILES
from src.settings import validate_metadata


def check_required_files(repo_root: Path):
    missing = []
    for rel in DOC_FILES + STATS_FILES + VIS_FILES:
        if not (repo_root / rel).exists():
            missing.append(rel)
    return missing


def main():
    parser = argparse.ArgumentParser(description="Validate dataset package completeness")
    parser.add_argument("--repo-root", default=".", help="Path to dataset repo root")
    parser.add_argument("--prepare", action="store_true", help="Run conversion preparation hook")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    if args.prepare:
        convert_and_prepare_dataset(repo_root)

    missing_meta = validate_metadata(repo_root)
    missing_files = check_required_files(repo_root)

    if missing_meta:
        print("Missing metadata fields:")
        for item in missing_meta:
            print(f" - {item}")
    if missing_files:
        print("Missing required files:")
        for item in missing_files:
            print(f" - {item}")

    if missing_meta or missing_files:
        raise SystemExit(1)

    print("Validation passed")


if __name__ == "__main__":
    main()
