#!/usr/bin/env python3
"""
Metadata Enricher
Auto-fills METADATA.json with computed statistics.
This ensures METADATA.json is always up-to-date with the actual dataset.
"""

import json
import yaml
import os
import re
from task_classification import resolve_task_classification


def _slugify_token(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def _country_aliases(country_name):
    base = _slugify_token(country_name)
    aliases = {base, base.replace("_", "")}
    parts = [p for p in base.split("_") if p]
    if parts:
        aliases.add(parts[0])
        aliases.add(parts[-1])
    if len(parts) >= 2:
        aliases.add("_".join(parts[:2]))
    return {a for a in aliases if a}


def _build_country_lookup(config):
    lookup = {}
    countries = config.get('countries', []) if isinstance(config, dict) else []
    for item in countries or []:
        if isinstance(item, dict):
            name = str(item.get('name', '')).strip()
        else:
            name = str(item).strip()
        if not name:
            continue
        cid = _slugify_token(name)
        if not cid:
            continue
        lookup[cid] = {
            'id': cid,
            'label': name,
            'aliases': _country_aliases(name),
        }
    return lookup


def _infer_country_id_from_split(split_name, country_lookup):
    split_slug = _slugify_token(split_name)
    if not split_slug:
        return None

    best_country = None
    best_len = -1
    for cid, meta in country_lookup.items():
        for alias in meta.get('aliases', []):
            if not alias:
                continue
            if re.search(rf"(^|_){re.escape(alias)}(_|$)", split_slug):
                if len(alias) > best_len:
                    best_country = cid
                    best_len = len(alias)
    if best_country:
        return best_country

    parts = [p for p in split_slug.split('_') if p]
    if len(parts) >= 2 and parts[-1] in {'train', 'test', 'val', 'valid', 'validation', 'dev'}:
        return "_".join(parts[:-1])
    if len(country_lookup) == 1:
        return next(iter(country_lookup.keys()))
    return split_slug


def _normalized_label(value):
    return _slugify_token(value).replace('_', '')


def _is_negative_class_label(name):
    n = _normalized_label(name)
    negative_tokens = (
        'non',
        'nocrack',
        'negative',
        'normal',
        'background',
        'bg',
        'intact',
        'healthy',
    )
    return any(tok in n for tok in negative_tokens)


def _is_positive_class_label(name):
    n = _normalized_label(name)
    if _is_negative_class_label(name):
        return False
    positive_tokens = (
        'crack',
        'damage',
        'defect',
        'distress',
        'pothole',
    )
    return any(tok in n for tok in positive_tokens)


def _resolve_positive_classes(config, observed_classes):
    explicit = config.get('positive_classes') or config.get('classification_positive_classes') or []
    explicit_norm = {_normalized_label(v) for v in explicit if str(v).strip()}
    if explicit_norm:
        return {
            cls for cls in observed_classes
            if _normalized_label(cls) in explicit_norm
        }

    guessed = {cls for cls in observed_classes if _is_positive_class_label(cls)}
    return guessed


def _compute_country_statistics(splits, config, task_type=''):
    if not isinstance(splits, dict) or not splits:
        return []

    country_lookup = _build_country_lookup(config)
    country_stats = {}
    task = str(task_type or '').strip().lower()
    is_classification = task == 'classification'

    observed_classes = set()
    if is_classification:
        for split_stats in (splits or {}).values():
            for cls_name in ((split_stats or {}).get('class_distribution', {}) or {}).keys():
                observed_classes.add(str(cls_name))
    positive_classes = _resolve_positive_classes(config, observed_classes) if is_classification else set()

    for split_name, split_stats in splits.items():
        cid = _infer_country_id_from_split(split_name, country_lookup)
        if not cid:
            continue

        label = country_lookup.get(cid, {}).get('label')
        if not label:
            label = cid.replace('_', ' ').title()

        rec = country_stats.setdefault(
            cid,
            {
                'name': label,
                'image_count': 0,
                'annotated_image_count': 0,
            },
        )

        split_payload = split_stats or {}
        split_num_images = int(split_payload.get('num_images', 0))
        split_num_annotated = int(split_payload.get('num_images_with_annotations', 0))

        if is_classification:
            class_distribution = split_payload.get('class_distribution', {}) or {}
            split_positive = sum(
                int(v)
                for k, v in class_distribution.items()
                if str(k) in positive_classes
            )
            if split_positive > 0:
                split_num_images = split_positive
                split_num_annotated = split_positive

        rec['image_count'] += split_num_images
        rec['annotated_image_count'] += split_num_annotated

    if not country_stats:
        return []

    ordered = []
    for cid in country_lookup.keys():
        if cid in country_stats:
            rec = country_stats[cid]
            rec['annotation_coverage'] = round(
                rec['annotated_image_count'] / max(1, rec['image_count']), 4
            )
            ordered.append(rec)

    remaining = sorted([cid for cid in country_stats.keys() if cid not in country_lookup])
    for cid in remaining:
        rec = country_stats[cid]
        rec['annotation_coverage'] = round(
            rec['annotated_image_count'] / max(1, rec['image_count']), 4
        )
        ordered.append(rec)

    return ordered


def enrich_metadata_from_stats(metadata_path, stats_data, config):
    """
    Enrich METADATA.json with computed statistics.
    
    Args:
        metadata_path: Path to METADATA.json
        stats_data: Computed stats from stats.json
        config: Config dict from config.yaml
    """
    
    # Load template metadata
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Extract computed stats
    global_stats = stats_data.get('global', {})
    splits = stats_data.get('splits', {})
    
    # Auto-fill basic statistics
    metadata['statistics'] = {
        'n_images': global_stats.get('num_images', 0),
        'n_annotations': global_stats.get('num_annotations', 0),
        'n_images_with_annotations': global_stats.get('num_images_with_annotations', 0),
        'average_annotations_per_image': (
            global_stats.get('num_annotations', 0) / max(1, global_stats.get('num_images', 1))
        ),
        'dataset_size': global_stats.get('dataset_size_human', '0 MB'),
        'dataset_size_bytes': int(global_stats.get('dataset_size_bytes', 0)),
        'dataset_file_count': int(global_stats.get('dataset_file_count', 0)),
        'image_format': config.get('format', 'coco'),
        'class_distribution': global_stats.get('class_distribution', {}),
        'split': {}
    }
    
    # Add split breakdown
    total_images = global_stats.get('num_images', 0)
    for split_name, split_stats in splits.items():
        n_images = split_stats.get('num_images', 0)
        percentage = (n_images / total_images) if total_images > 0 else 0
        
        metadata['statistics']['split'][split_name] = {
            'n_images': n_images,
            'percentage': round(percentage, 2),
            'n_images_with_annotations': int(split_stats.get('num_images_with_annotations', 0)),
            'annotation_coverage': round(float(split_stats.get('annotation_coverage', 0)), 4),
            'dataset_size': split_stats.get('dataset_size_human', '0 MB'),
            'dataset_size_bytes': int(split_stats.get('dataset_size_bytes', 0)),
            'dataset_file_count': int(split_stats.get('dataset_file_count', 0)),
        }
    
    # Update basic info from config while preserving optional fields.
    basic_info = metadata.get('basic_info', {})
    basic_info['id'] = config.get('dataset_id', basic_info.get('id', 'unknown'))
    basic_info['name'] = config.get('dataset_name', basic_info.get('name', 'Unknown Dataset'))
    basic_info['short_name'] = basic_info.get('short_name') or config.get('dataset_id', 'dataset')
    basic_info['description'] = basic_info.get('description') or config.get('description', 'Road damage dataset package')
    basic_info['year'] = config.get('year', basic_info.get('year', 2024))
    metadata['basic_info'] = basic_info
    
    project_context = metadata.get('project_context', {})
    task_value = config.get('task_type', project_context.get('task', 'object_detection'))
    project_context['task'] = task_value
    # Keep task list aligned with current package task and avoid stale template defaults.
    project_context['tasks'] = [task_value]

    task_classification = resolve_task_classification(config)
    project_context['task_classification'] = task_classification
    if task_classification.get('primary_task') == 'segmentation':
        seg_type = task_classification.get('segmentation_type')
        if seg_type:
            project_context['segmentation_type'] = seg_type
            project_context['task_subtype'] = seg_type
    metadata['project_context'] = project_context
    
    # Update geographic info
    geo = metadata.get('geographic_coverage', {})
    computed_countries = _compute_country_statistics(splits, config, config.get('task_type', ''))
    if computed_countries:
        geo['countries'] = computed_countries
    else:
        geo['countries'] = config.get('countries', geo.get('countries', []))
    geo['temporal_scope'] = config.get('temporal_scope', geo.get('temporal_scope', '2021-2024'))
    metadata['geographic_coverage'] = geo
    
    # Update authors
    if 'authors' in config:
        metadata['authors'] = config['authors']
    
    # Update license and links
    if 'license' in config:
        metadata['license'] = config['license']

    links = metadata.get('links', {})
    cfg_links = config.get('links', {})
    links.update(cfg_links)
    dataset_id = metadata['basic_info']['id']
    links.setdefault('github', f"https://github.com/large-road-damage-datalake/{dataset_id}")
    links.setdefault(
        'download',
        f"https://github.com/large-road-damage-datalake/{dataset_id}/releases/latest"
    )
    metadata['links'] = links
    
    # Update citation
    citation = metadata.get('citation', {})
    if 'citations' in config and 'bibtex' in config['citations']:
        citation['bibtex'] = config['citations']['bibtex']
    citation.setdefault(
        'bibtex',
        (
            f"@dataset{{{dataset_id},\n"
            f"  title={{{metadata['basic_info']['name']}}},\n"
            "  author={Unknown},\n"
            f"  year={{{metadata['basic_info']['year']}}}\n"
            "}"
        )
    )
    metadata['citation'] = citation
    
    # Save enriched metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Enriched METADATA.json")
    print(f"  - Images: {metadata['statistics']['n_images']}")
    print(f"  - Annotations: {metadata['statistics']['n_annotations']}")
    print(f"  - Classes: {len(metadata['statistics']['class_distribution'])}")


def validate_metadata(metadata_path):
    """
    Validate that METADATA.json has all required fields.
    Returns tuple: (is_valid, missing_fields)
    """
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    required_fields = [
        'basic_info.id',
        'basic_info.name',
        'project_context.task',
        'statistics.n_images',
        'license',
        'citation'
    ]
    
    missing = []
    for field_path in required_fields:
        parts = field_path.split('.')
        obj = metadata
        for part in parts:
            if part not in obj:
                missing.append(field_path)
                break
            obj = obj[part]
    
    return len(missing) == 0, missing


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 3:
        stats_file = sys.argv[2]
        config_file = sys.argv[3]
        
        with open(stats_file, 'r') as f:
            stats_data = json.load(f)
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        enrich_metadata_from_stats(sys.argv[1], stats_data, config)
    else:
        print("Usage: python enrich_metadata.py <metadata_path> <stats_path> <config_path>")
