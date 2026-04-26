#!/usr/bin/env python3
"""
Visualization Generator
Creates visual assets for dataset discovery:
- Class distribution charts
- Image samples collection
- Dataset statistics plots
- Annotation type distribution
"""

import os
import json
import random
from pathlib import Path
from collections import defaultdict
import yaml


def generate_visualizations(config_path, output_dir, stats_data):
    """
    Generate all visualizations for the dataset.
    
    Args:
        config_path: Path to config.yaml
        output_dir: Output directory where visualizations will be saved
        stats_data: Computed stats from stats.json
    """
    
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Generate class distribution chart data
    generate_class_distribution_json(stats_data, vis_dir)
    
    # Generate dataset summary stats
    generate_dataset_summary(stats_data, vis_dir)
    
    # Generate split distribution
    generate_split_distribution(stats_data, vis_dir)
    
    sample_count = count_available_samples(config_path)
    generate_visualizations_manifest(vis_dir, stats_data, sample_count)
    
    print(f"✓ Visualization data generated in {vis_dir}")


def generate_class_distribution_json(stats_data, vis_dir):
    """
    Generate class distribution data for charts.
    Output: visualizations/class_distribution.json
    """
    
    global_stats = stats_data.get('global', {})
    class_dist = global_stats.get('class_distribution', {})
    
    chart_data = {
        'type': 'bar',
        'title': 'Class Distribution',
        'data': {
            'labels': list(class_dist.keys()),
            'values': list(class_dist.values()),
            'total': sum(class_dist.values())
        }
    }
    
    # Calculate percentages
    total = chart_data['data']['total']
    if total > 0:
        chart_data['data']['percentages'] = [
            round(v / total * 100, 2) for v in chart_data['data']['values']
        ]
    
    output_file = os.path.join(vis_dir, 'class_distribution.json')
    with open(output_file, 'w') as f:
        json.dump(chart_data, f, indent=2)
    
    print(f"  ✓ Generated: class_distribution.json")


def generate_dataset_summary(stats_data, vis_dir):
    """
    Generate overall dataset summary statistics.
    Output: visualizations/dataset_summary.json
    """
    
    global_stats = stats_data.get('global', {})
    splits = stats_data.get('splits', {})
    resolution_hist = global_stats.get('image_resolution_histogram', {}) or {}
    
    summary = {
        'total_images': global_stats.get('num_images', 0),
        'total_annotations': global_stats.get('num_annotations', 0),
        'total_classes': global_stats.get('num_classes', 0),
        'annotated_images': global_stats.get('num_images_with_annotations', 0),
        'annotation_coverage': global_stats.get('annotation_coverage', 0),
        'image_resolution_histogram': resolution_hist,
        'resolution_details': [
            {
                'resolution': str(key),
                'count': int(value)
            }
            for key, value in resolution_hist.items()
        ],
        'splits_breakdown': {}
    }
    
    # Breakdown by split
    for split_name, split_stats in splits.items():
        summary['splits_breakdown'][split_name] = {
            'images': split_stats.get('num_images', 0),
            'annotations': split_stats.get('num_annotations', 0),
            'annotated_images': split_stats.get('num_images_with_annotations', 0),
            'annotation_coverage': split_stats.get('annotation_coverage', 0),
            'classes': split_stats.get('class_distribution', {}),
            'image_resolution_histogram': split_stats.get('image_resolution_histogram', {}) or {},
        }
    
    output_file = os.path.join(vis_dir, 'dataset_summary.json')
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  ✓ Generated: dataset_summary.json")


def generate_split_distribution(stats_data, vis_dir):
    """
    Generate split distribution chart data.
    Output: visualizations/split_distribution.json
    """
    
    splits = stats_data.get('splits', {})
    
    chart_data = {
        'type': 'pie',
        'title': 'Dataset Split Distribution',
        'data': {
            'labels': [],
            'values': [],
            'colors': ['#3498db', '#e74c3c', '#2ecc71']  # train, val, test colors
        }
    }
    
    total_images = 0
    split_counts = {}
    
    for split_name, split_stats in splits.items():
        count = split_stats.get('num_images', 0)
        split_counts[split_name] = count
        total_images += count
    
    # Add to chart in order: train, val, test
    order = ['train', 'val', 'test']
    for split_name in order:
        if split_name in split_counts:
            count = split_counts[split_name]
            percentage = (count / total_images * 100) if total_images > 0 else 0
            
            chart_data['data']['labels'].append(f"{split_name.capitalize()} ({percentage:.1f}%)")
            chart_data['data']['values'].append(count)
    
    output_file = os.path.join(vis_dir, 'split_distribution.json')
    with open(output_file, 'w') as f:
        json.dump(chart_data, f, indent=2)
    
    print(f"  ✓ Generated: split_distribution.json")


def count_available_samples(config_path):
    """Count candidate sample images from split roots in config."""
    if not os.path.exists(config_path):
        return 0

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    cfg_dir = os.path.dirname(os.path.abspath(config_path))
    splits = cfg.get('splits') or ({'full': cfg.get('data')} if cfg.get('data') else {})
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    count = 0

    for _, split_cfg in splits.items():
        if not split_cfg:
            continue
        root = split_cfg.get('images_root')
        if not root:
            continue
        if not os.path.isabs(root):
            root = os.path.join(cfg_dir, root)
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            if os.path.splitext(name)[1].lower() in exts:
                count += 1

    return count


def generate_visualizations_manifest(vis_dir, stats_data, sample_count):
    """
    Generate manifest file listing all available visualizations.
    This helps the website know what's available.
    Output: visualizations/manifest.json
    """
    
    manifest = {
        'generated_at': Path(vis_dir).stat().st_mtime if os.path.exists(vis_dir) else 0,
        'visualizations': [
            {
                'name': 'class_distribution',
                'type': 'bar_chart',
                'file': 'class_distribution.json',
                'description': 'Distribution of annotations by class'
            },
            {
                'name': 'dataset_summary',
                'type': 'summary_stats',
                'file': 'dataset_summary.json',
                'description': 'Overall dataset statistics'
            },
            {
                'name': 'split_distribution',
                'type': 'pie_chart',
                'file': 'split_distribution.json',
                'description': 'Train/Val/Test split percentages'
            }
        ],
        'sample_images': {
            'location': 'samples/',
            'count': sample_count,
            'status': 'available' if sample_count > 0 else 'missing'
        }
    }
    
    output_file = os.path.join(vis_dir, 'manifest.json')
    with open(output_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"  ✓ Generated: manifest.json")


if __name__ == "__main__":
    # Example usage (for testing)
    import sys
    if len(sys.argv) > 2:
        generate_visualizations(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python generate_visualizations.py <config_path> <output_dir>")
