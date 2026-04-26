#!/usr/bin/env python3
"""
Main Pipeline Orchestrator
Runs the complete dataset processing pipeline:
1. Validate config
2. Compute stats
3. Generate visualizations
4. Enrich METADATA.json
5. Optional: Upload to GitHub
"""

import argparse
import yaml
import json
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

os.environ.setdefault("MPLBACKEND", "Agg")

# Import local modules
from generate_visualizations import generate_visualizations
from enrich_metadata import enrich_metadata_from_stats
from build_artifacts import build_artifacts
from validate_dataset_package import print_validation_report
from task_classification import resolve_task_classification


def _format_size(num_bytes):
    """Return human-readable size in MB/GB."""
    if num_bytes <= 0:
        return "0 MB"
    gb = 1024 ** 3
    mb = 1024 ** 2
    if num_bytes >= gb:
        return f"{num_bytes / gb:.2f} GB"
    return f"{num_bytes / mb:.2f} MB"


def _collect_split_files(split_cfg):
    """Collect all source files referenced by a split config."""
    files = set()
    if not isinstance(split_cfg, dict):
        return files

    for key in ["images_root", "annotations", "masks_root"]:
        p = split_cfg.get(key)
        if not p:
            continue
        p = os.path.normpath(str(p))
        if os.path.isfile(p):
            files.add(os.path.normcase(p))
        elif os.path.isdir(p):
            for root, _, fnames in os.walk(p):
                for name in fnames:
                    files.add(os.path.normcase(os.path.join(root, name)))

    return files


def _sum_file_sizes(file_paths):
    total = 0
    for p in file_paths:
        try:
            total += int(os.path.getsize(p))
        except OSError:
            continue
    return total


def validate_config(config_path):
    """Validate config.yaml exists and has required fields."""
    if not os.path.exists(config_path):
        print(f"❌ Error: Config file '{config_path}' not found.")
        return False

    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config = yaml.safe_load(f) or {}

    # Normalize potential BOM in first key.
    config = {str(k).lstrip('\ufeff'): v for k, v in config.items()}
    
    required_fields = ['dataset_id', 'dataset_name', 'task_type', 'format']
    for field in required_fields:
        if field not in config:
            print(f"❌ Error: Missing required field in config: '{field}'")
            return False
    
    supported_formats = ['coco', 'yolo', 'voc', 'image_folder', 'png_masks']
    if config['format'] not in supported_formats:
        print(f"❌ Error: Unsupported format '{config['format']}'")
        return False
    
    return True


def create_output_structure(output_dir):
    """Create the standardized dataset repo structure."""
    directories = [
        'data',
        'stats',
        'visualizations',
        'visualizations/samples',
        'annotations',
        'src',
    ]
    
    for d in directories:
        path = os.path.join(output_dir, d)
        os.makedirs(path, exist_ok=True)
    
    print(f"✓ Created directory structure in {output_dir}")


def copy_template_files(output_dir):
    """Copy template docs and scaffold into output package."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.normpath(os.path.join(script_dir, '..', 'template')),
        os.path.normpath(os.path.join(script_dir, '..')),
    ]

    template_dir = None
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, 'METADATA.json')) and os.path.isdir(os.path.join(cand, 'src')):
            template_dir = cand
            break

    if not template_dir:
        raise FileNotFoundError(
            "Could not resolve template root. Expected METADATA.json and src/ near scripts/."
        )

    template_files = [
        '.gitignore',
        'ABSTRACT.md',
        'CITATION.bib',
        'CITATION.md',
        'DOWNLOAD.md',
        'LICENSE.md',
        'METADATA.json',
        'README.md',
        'SUMMARY.md',
        'requirements.txt',
        'create_venv.sh',
        'local.env',
        'config.example.yaml',
    ]

    for file in template_files:
        src = os.path.join(template_dir, file)
        dst = os.path.join(output_dir, file)
        if os.path.exists(src):
            shutil.copy(src, dst)

    # Copy src package and helper scripts in the generated dataset package.
    src_src = os.path.join(template_dir, 'src')
    src_dst = os.path.join(output_dir, 'src')
    if os.path.isdir(src_src):
        shutil.copytree(src_src, src_dst, dirs_exist_ok=True)

    # Keep computational scripts in package for reproducibility.
    scripts_src = os.path.join(template_dir, 'scripts')
    scripts_dst = os.path.join(output_dir, 'scripts')
    if os.path.isdir(scripts_src):
        shutil.copytree(scripts_src, scripts_dst, dirs_exist_ok=True)

    print(f"✓ Copied template files")


def _resolve_split_paths(config, config_dir):
    """Resolve relative paths in config to absolute paths for processing."""
    cfg = dict(config)
    splits = cfg.get('splits')
    data = cfg.get('data')

    def _resolve_split(split_cfg):
        resolved = dict(split_cfg)
        for key in ['annotations', 'images_root', 'masks_root']:
            p = resolved.get(key)
            if p and not os.path.isabs(p):
                resolved[key] = os.path.normpath(os.path.join(config_dir, p))
        return resolved

    if splits:
        cfg['splits'] = {name: _resolve_split(scfg) for name, scfg in splits.items()}
    elif data:
        cfg['data'] = _resolve_split(data)

    return cfg


def run_stats_computation(config_path, output_dir):
    """Run statistics computation via compute_stats.py."""
    from stats_core import compute_split_stats, aggregate_global_stats
    
    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)
    config = {str(k).lstrip('\ufeff'): v for k, v in config.items()}

    config_dir = os.path.dirname(os.path.abspath(config_path))
    config = _resolve_split_paths(config, config_dir)
    
    task_type = config.get('task_type')
    fmt = config.get('format')
    task_classification = resolve_task_classification(config)
    splits = config.get('splits') or {'data': config.get('data')}
    
    all_stats = {}
    all_source_files = set()
    print(f"\n📊 Computing statistics for {config.get('dataset_name')}...")
    
    for split_name, split_cfg in splits.items():
        print(f"  Processing split: {split_name}...")
        effective_split_cfg = dict(split_cfg or {})
        if 'class_map' not in effective_split_cfg and 'class_map' in config:
            effective_split_cfg['class_map'] = config.get('class_map')
        if 'class_exclude' not in effective_split_cfg and 'class_exclude' in config:
            effective_split_cfg['class_exclude'] = config.get('class_exclude')

        stats = compute_split_stats(task_type, fmt, effective_split_cfg)

        split_files = _collect_split_files(effective_split_cfg)
        split_size_bytes = _sum_file_sizes(split_files)
        stats['dataset_size_bytes'] = split_size_bytes
        stats['dataset_size_human'] = _format_size(split_size_bytes)
        stats['dataset_file_count'] = len(split_files)
        all_source_files.update(split_files)

        all_stats[split_name] = stats
    
    print("  Aggregating global stats...")
    global_stats = aggregate_global_stats(all_stats)
    global_size_bytes = _sum_file_sizes(all_source_files)
    global_stats['dataset_size_bytes'] = global_size_bytes
    global_stats['dataset_size_human'] = _format_size(global_size_bytes)
    global_stats['dataset_file_count'] = len(all_source_files)
    
    final_output = {
        'dataset_name': config.get('dataset_name'),
        'dataset_id': config.get('dataset_id'),
        'task_type': task_type,
        'task_classification': task_classification,
        'format': fmt,
        'computed_at': datetime.now().isoformat(),
        'global': global_stats,
        'splits': all_stats
    }
    
    stats_file = os.path.join(output_dir, 'stats', 'stats.json')
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    
    with open(stats_file, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print(f"✓ Stats written to {stats_file}\n")
    return final_output


def main():
    parser = argparse.ArgumentParser(
        description="Run complete dataset processing pipeline"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Skip visualization generation (faster for testing)"
    )
    parser.add_argument(
        "--push-github",
        action="store_true",
        help="Push to GitHub after processing"
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Disable strict package validation gate"
    )
    
    args = parser.parse_args()
    
    # Step 1: Validate config
    print("=" * 60)
    print("🚀 LARGE ROAD DAMAGE DATALAKE - PIPELINE ORCHESTRATOR")
    print("=" * 60)
    print("\n1️⃣  Validating configuration...")
    
    if not validate_config(args.config):
        sys.exit(1)
    
    with open(args.config, 'r', encoding='utf-8-sig') as f:
        config = yaml.safe_load(f) or {}
    config = {str(k).lstrip('\ufeff'): v for k, v in config.items()}
    config['_config_dir'] = os.path.dirname(os.path.abspath(args.config))
    
    print(f"✓ Config valid: {config['dataset_name']}")
    
    # Step 2: Create output structure
    print(f"\n2️⃣  Creating output structure...")
    create_output_structure(args.output)
    
    # Step 3: Copy template files
    print(f"\n3️⃣  Copying template files...")
    copy_template_files(args.output)
    
    # Step 4: Compute statistics
    print(f"\n4️⃣  Computing statistics...")
    stats_data = run_stats_computation(args.config, args.output)
    
    # Step 5: Generate visualizations
    if not args.skip_visualizations:
        print(f"5️⃣  Generating visualizations...")
        try:
            generate_visualizations(args.config, args.output, stats_data)
            print(f"✓ Visualizations generated")
        except Exception as e:
            print(f"⚠️  Warning: Visualization generation failed: {e}")
    else:
        print(f"5️⃣  Skipping visualizations (--skip-visualizations)")
    
    # Step 6: Enrich metadata
    print(f"\n6️⃣  Enriching METADATA.json...")
    try:
        metadata_path = os.path.join(args.output, 'METADATA.json')
        enrich_metadata_from_stats(metadata_path, stats_data, config)
        print(f"✓ METADATA.json enriched")
    except Exception as e:
        print(f"⚠️  Warning: Metadata enrichment failed: {e}")

    # Step 7: Build datasetninja-style artifacts (rich stats + previews)
    print(f"\n7️⃣  Building datasetninja-style artifacts...")
    try:
        build_artifacts(config, args.output, stats_data)
    except Exception as e:
        print(f"⚠️  Warning: Artifact build failed: {e}")

    # Step 8: Strict validation gate
    if not args.no_strict:
        print(f"\n8️⃣  Running strict package validation...")
        valid = print_validation_report(args.output)
        if not valid:
            print("\n❌ Package is not publication-ready. Fix validation errors and rerun.")
            sys.exit(2)
    
    # Step 9: GitHub push (optional)
    if args.push_github:
        print(f"\n9️⃣  Pushing to GitHub...")
        print("❌ GitHub push not yet implemented. Manual GitHub setup required.")
        print("   See: https://github.com/large-road-damage-datalake/")
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"📁 Output directory: {args.output}")
    print(f"📊 Dataset: {config['dataset_name']}")
    print(f"🖼️  Stats: {os.path.join(args.output, 'stats', 'stats.json')}")
    print(f"🎨 Visualizations: {os.path.join(args.output, 'visualizations')}")
    print(f"📄 Metadata: {os.path.join(args.output, 'METADATA.json')}")
    print("\nNext steps:")
    print("1. Review METADATA.json and update any missing fields manually")
    print("2. Review statistics and visualizations")
    print("3. Create GitHub repo in organization")
    print("4. Push to GitHub: git push origin main")
    print("=" * 60)


if __name__ == "__main__":
    main()
