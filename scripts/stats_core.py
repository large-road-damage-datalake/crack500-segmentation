from collections import defaultdict
import loaders
import re
import statistics


def _safe_stats(values):
    if not values:
        return {"mean": 0, "median": 0}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


def _histogram(values):
    hist = defaultdict(int)
    for v in values:
        hist[str(v)] += 1
    return dict(sorted(hist.items(), key=lambda kv: int(kv[0])))


def _resolution_histogram(widths, heights):
    hist = defaultdict(int)
    for w, h in zip(widths or [], heights or []):
        try:
            wi = int(round(float(w)))
            hi = int(round(float(h)))
        except Exception:
            continue
        if wi <= 0 or hi <= 0:
            continue
        hist[f"{wi}x{hi}"] += 1

    def _sort_key(item):
        key = str(item[0])
        m = re.match(r"^(\d+)x(\d+)$", key)
        if not m:
            return (10**9, 10**9)
        return (int(m.group(1)), int(m.group(2)))

    return dict(sorted(hist.items(), key=_sort_key))

def compute_split_stats(task_type, fmt, split_cfg):
    """
    Compute stats for a single split.
    """
    images_root = split_cfg.get('images_root', '')
    annotations = split_cfg.get('annotations', '') # file or folder
    masks_root = split_cfg.get('masks_root', '')
    mask_suffixes = split_cfg.get('mask_suffixes', [''])
    class_map = split_cfg.get('class_map', {})
    class_exclude = split_cfg.get('class_exclude', [])
    connected_components = split_cfg.get('connected_components', True)
    min_component_area_px = split_cfg.get('min_component_area_px', 1)
    
    # Dispatch
    data = None
    if fmt == 'coco':
        data = loaders.load_coco_stats(annotations, images_root)
    elif fmt == 'yolo':
        data = loaders.load_yolo_stats(
            images_root,
            annotations,
            class_map=class_map,
            class_exclude=class_exclude,
        )
    elif fmt == 'voc':
        data = loaders.load_voc_stats(
            images_root,
            annotations,
            class_map=class_map,
            class_exclude=class_exclude,
        )
    elif fmt == 'image_folder':
        data = loaders.load_image_folder_stats(
            images_root,
            class_map=class_map,
            class_exclude=class_exclude,
        )
    elif fmt == 'png_masks':
        data = loaders.load_png_mask_stats(
            images_root,
            masks_root,
            class_map=class_map,
            class_exclude=class_exclude,
            mask_suffixes=mask_suffixes,
            connected_components=connected_components,
            min_component_area_px=min_component_area_px,
        )
    else:
        raise ValueError(f"Unknown format: {fmt}")
        
    if data is None:
        # Return empty stats if path failed
        return {
            'num_images': 0,
            'num_annotations': 0,
            'num_images_with_annotations': 0,
            'annotation_coverage': 0,
            'class_distribution': {},
            'objects_per_image': {'mean': 0, 'median': 0},
            'image_resolution': {'mean_width': 0, 'mean_height': 0},
            'bbox_area_rel': {'mean': 0, 'median': 0}
        }
    
    # Post-process raw data into summary stats
    summary = {}
    summary['num_images'] = data['num_images']
    summary['num_annotations'] = data['num_annotations']
    summary['class_distribution'] = dict(data['class_distribution'])
    
    # Objects per image
    objs = data.get('objects_per_image', [])
    if objs:
        summary['objects_per_image'] = {
            'mean': statistics.mean(objs),
            'median': statistics.median(objs)
        }
        summary['objects_per_image_hist'] = _histogram(objs)
    else:
        summary['objects_per_image'] = {'mean': 0, 'median': 0}
        summary['objects_per_image_hist'] = {}

    images_with_annotations = sum(1 for v in objs if v and v > 0)
    summary['num_images_with_annotations'] = images_with_annotations
    summary['annotation_coverage'] = (
        images_with_annotations / max(1, summary['num_images'])
    )
        
    # Resolution
    widths = data['image_resolution']['widths']
    heights = data['image_resolution']['heights']
    if widths and heights:
        summary['image_resolution'] = {
            'mean_width': statistics.mean(widths),
            'mean_height': statistics.mean(heights),
        }
        summary['image_resolution_histogram'] = _resolution_histogram(widths, heights)
    else:
        summary['image_resolution'] = {'mean_width': 0, 'mean_height': 0}
        summary['image_resolution_histogram'] = {}
        
    # Bbox area
    areas = data.get('bbox_area_rel', [])
    if areas:
        summary['bbox_area_rel'] = {
            'mean': statistics.mean(areas),
            'median': statistics.median(areas)
        }
    else:
        summary['bbox_area_rel'] = {'mean': 0, 'median': 0}

    # Sizes by class (mean/median area fraction of image)
    by_class = data.get('bbox_area_rel_by_class', {})
    summary['class_sizes'] = {
        cls: _safe_stats(vs)
        for cls, vs in by_class.items()
    }

    # Keep raw slices for artifact builders.
    summary['_raw'] = {
        'objects_per_image': objs,
        'bbox_area_rel': areas,
        'class_sets_per_image': data.get('class_sets_per_image', []),
        'class_counts_per_image': data.get('class_counts_per_image', []),
        'class_area_sum_per_image': data.get('class_area_sum_per_image', []),
        'image_paths': data.get('image_paths', []),
        'bbox_area_rel_by_class': {
            cls: vals for cls, vals in by_class.items()
        },
        'bbox_shapes_by_class': {
            cls: vals for cls, vals in data.get('bbox_shapes_by_class', {}).items()
        },
    }
    
    # Classification fractions
    if task_type == 'classification':
        total = max(1, summary['num_images'])
        summary['class_fractions'] = {k: v/total for k,v in summary['class_distribution'].items()}
        
    return summary

def aggregate_global_stats(split_stats_dict):
    """
    Aggregate stats across splits.
    """
    global_stats = {
        'num_images': 0,
        'num_annotations': 0,
        'num_images_with_annotations': 0,
        'class_distribution': defaultdict(int),
        'image_resolution_histogram': defaultdict(int),
    }
    
    for split_name, stats in split_stats_dict.items():
        global_stats['num_images'] += stats['num_images']
        global_stats['num_annotations'] += stats['num_annotations']
        global_stats['num_images_with_annotations'] += int(stats.get('num_images_with_annotations', 0))
        for k, v in stats['class_distribution'].items():
            global_stats['class_distribution'][k] += v
        for res_key, cnt in (stats.get('image_resolution_histogram', {}) or {}).items():
            global_stats['image_resolution_histogram'][str(res_key)] += int(cnt)
            
    # Convert defaultdict to dict
    global_stats['class_distribution'] = dict(global_stats['class_distribution'])
    global_stats['image_resolution_histogram'] = dict(
        sorted(
            global_stats['image_resolution_histogram'].items(),
            key=lambda kv: (
                int(kv[0].split('x')[0]) if 'x' in kv[0] and kv[0].split('x')[0].isdigit() else 10**9,
                int(kv[0].split('x')[1]) if 'x' in kv[0] and kv[0].split('x')[1].isdigit() else 10**9,
            ),
        )
    )
    global_stats['num_classes'] = len(global_stats['class_distribution'])
    
    # Imbalance
    counts = list(global_stats['class_distribution'].values())
    if counts:
        max_c = max(counts)
        min_c = min(counts)
        global_stats['imbalance'] = {
            'max_class_count': max_c,
            'min_class_count': min_c,
            'ratio': max_c / max(1, min_c)
        }
    else:
        global_stats['imbalance'] = {'max': 0, 'min': 0, 'ratio': 0}

    # Merge raw views across splits for richer artifacts.
    merged_raw = {
        'objects_per_image': [],
        'bbox_area_rel': [],
        'class_sets_per_image': [],
        'class_counts_per_image': [],
        'class_area_sum_per_image': [],
        'image_paths': [],
        'bbox_area_rel_by_class': defaultdict(list),
        'bbox_shapes_by_class': defaultdict(list),
    }
    for _, split_stats in split_stats_dict.items():
        raw = split_stats.get('_raw', {})
        merged_raw['objects_per_image'].extend(raw.get('objects_per_image', []))
        merged_raw['bbox_area_rel'].extend(raw.get('bbox_area_rel', []))
        merged_raw['class_sets_per_image'].extend(raw.get('class_sets_per_image', []))
        merged_raw['class_counts_per_image'].extend(raw.get('class_counts_per_image', []))
        merged_raw['class_area_sum_per_image'].extend(raw.get('class_area_sum_per_image', []))
        merged_raw['image_paths'].extend(raw.get('image_paths', []))
        for cls, vals in raw.get('bbox_area_rel_by_class', {}).items():
            merged_raw['bbox_area_rel_by_class'][cls].extend(vals)
        for cls, vals in raw.get('bbox_shapes_by_class', {}).items():
            merged_raw['bbox_shapes_by_class'][cls].extend(vals)

    global_stats['_raw'] = {
        'objects_per_image': merged_raw['objects_per_image'],
        'bbox_area_rel': merged_raw['bbox_area_rel'],
        'class_sets_per_image': merged_raw['class_sets_per_image'],
        'class_counts_per_image': merged_raw['class_counts_per_image'],
        'class_area_sum_per_image': merged_raw['class_area_sum_per_image'],
        'image_paths': merged_raw['image_paths'],
        'bbox_area_rel_by_class': dict(merged_raw['bbox_area_rel_by_class']),
        'bbox_shapes_by_class': dict(merged_raw['bbox_shapes_by_class']),
    }
    if not global_stats['num_images_with_annotations'] and merged_raw['objects_per_image']:
        global_stats['num_images_with_annotations'] = sum(
            1 for v in merged_raw['objects_per_image'] if v and v > 0
        )
    global_stats['annotation_coverage'] = (
        global_stats['num_images_with_annotations'] / max(1, global_stats['num_images'])
    )
    global_stats['objects_per_image_hist'] = _histogram(merged_raw['objects_per_image'])
    global_stats['class_sizes'] = {
        cls: _safe_stats(vals)
        for cls, vals in merged_raw['bbox_area_rel_by_class'].items()
    }
        
    return global_stats
