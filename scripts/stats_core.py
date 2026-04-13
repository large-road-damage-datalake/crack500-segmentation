from collections import defaultdict
import loaders
import statistics

def compute_split_stats(task_type, fmt, split_cfg):
    """
    Compute stats for a single split.
    """
    images_root = split_cfg.get('images_root', '')
    annotations = split_cfg.get('annotations', '') # file or folder
    masks_root = split_cfg.get('masks_root', '')
    
    # Dispatch
    data = None
    if fmt == 'coco':
        data = loaders.load_coco_stats(annotations, images_root)
    elif fmt == 'yolo':
        data = loaders.load_yolo_stats(images_root, annotations)
    elif fmt == 'voc':
        data = loaders.load_voc_stats(images_root, annotations)
    elif fmt == 'image_folder':
        data = loaders.load_image_folder_stats(images_root)
    elif fmt == 'png_masks':
        data = loaders.load_png_mask_stats(images_root, masks_root)
    else:
        raise ValueError(f"Unknown format: {fmt}")
        
    if data is None:
        # Return empty stats if path failed
        return {
            'num_images': 0,
            'num_annotations': 0,
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
    else:
        summary['objects_per_image'] = {'mean': 0, 'median': 0}
        
    # Resolution
    widths = data['image_resolution']['widths']
    heights = data['image_resolution']['heights']
    if widths and heights:
        summary['image_resolution'] = {
            'mean_width': statistics.mean(widths),
            'mean_height': statistics.mean(heights),
            # could add median if wanted
        }
    else:
        summary['image_resolution'] = {'mean_width': 0, 'mean_height': 0}
        
    # Bbox area
    areas = data.get('bbox_area_rel', [])
    if areas:
        summary['bbox_area_rel'] = {
            'mean': statistics.mean(areas),
            'median': statistics.median(areas)
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
        'class_distribution': defaultdict(int)
    }
    
    for split_name, stats in split_stats_dict.items():
        global_stats['num_images'] += stats['num_images']
        global_stats['num_annotations'] += stats['num_annotations']
        for k, v in stats['class_distribution'].items():
            global_stats['class_distribution'][k] += v
            
    # Convert defaultdict to dict
    global_stats['class_distribution'] = dict(global_stats['class_distribution'])
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
        
    return global_stats
