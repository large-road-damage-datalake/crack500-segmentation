import argparse
import yaml
import json
import os
import sys
from collections import OrderedDict
from stats_core import compute_split_stats, aggregate_global_stats

def main():
    parser = argparse.ArgumentParser(description="Compute dataset statistics.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--out", default="stats/stats.json", help="Output path for stats json")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.")
        sys.exit(1)
        
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    task_type = config.get('task_type')
    fmt = config.get('format')
    
    if task_type not in ['object_detection', 'segmentation', 'classification']:
        print(f"Error: Unsupported task_type '{task_type}'")
        sys.exit(1)
        
    supported_formats = ['coco', 'yolo', 'voc', 'image_folder', 'png_masks']
    if fmt not in supported_formats:
        print(f"Error: Unsupported format '{fmt}'")
        sys.exit(1)
        
    # Check split vs data
    splits = config.get('splits')
    data = config.get('data')
    
    if splits and data:
        print("Error: Cannot define both 'splits' and 'data' in config.")
        sys.exit(1)
    if not splits and not data:
        print("Error: Must define either 'splits' or 'data' in config.")
        sys.exit(1)
        
    if data:
        # Treat as synthetic 'full' split
        splits = {'full': data}
        
    all_stats = {}
    print(f"Computing stats for {config.get('dataset_name', 'Dataset')} ({task_type}/{fmt})...")
    
    for split_name, split_cfg in splits.items():
        print(f"Processing split: {split_name}...")
        stats = compute_split_stats(task_type, fmt, split_cfg)
        all_stats[split_name] = stats
        
    print("Aggregating global stats...")
    global_stats = aggregate_global_stats(all_stats)
    
    final_output = {
        'dataset_name': config.get('dataset_name'),
        'task_type': task_type,
        'format': fmt,
        'global': global_stats,
        'splits': all_stats
    }
    
    # Ensure stats dir exists
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    with open(args.out, 'w') as f:
        json.dump(final_output, f, indent=2)
        
    print(f"Done. Stats written to {args.out}")

if __name__ == "__main__":
    main()
