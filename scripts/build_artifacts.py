#!/usr/bin/env python3
"""
Build datasetninja-style publication artifacts.
Generates rich stats files and preview images for each dataset package.
"""

import json
import math
import os
import random
import re
import shutil
import statistics
import copy
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

try:
    from select_diverse_samples import copy_selected_samples, select_diverse_samples
except Exception:  # pragma: no cover
    copy_selected_samples = None
    select_diverse_samples = None


def _safe_makedirs(path):
    os.makedirs(path, exist_ok=True)


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _safe_stats(values):
    if not values:
        return {"mean": 0, "median": 0}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


def _extract_global_raw(stats_data):
    global_stats = stats_data.get("global", {})
    return global_stats.get("_raw", {})


def _sorted_classes(stats_data):
    class_dist = stats_data.get("global", {}).get("class_distribution", {})
    return [k for k, _ in sorted(class_dist.items(), key=lambda kv: kv[1], reverse=True)]


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


def _region_aliases(region_name):
    base = _slugify_token(region_name)
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
    countries = config.get("countries", []) if isinstance(config, dict) else []
    for item in countries or []:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
        else:
            name = str(item).strip()
        if not name:
            continue
        cid = _slugify_token(name)
        if not cid:
            continue
        aliases = _country_aliases(name)
        if isinstance(item, dict):
            extra_aliases = item.get("aliases", [])
            if isinstance(extra_aliases, list):
                aliases.update(_slugify_token(v) for v in extra_aliases if str(v).strip())
        lookup[cid] = {
            "id": cid,
            "label": name,
            "aliases": {a for a in aliases if a},
        }
    return lookup


def _build_region_lookup(config):
    lookup = {}
    regions = config.get("regions", []) if isinstance(config, dict) else []
    for item in regions or []:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
        else:
            name = str(item).strip()
        if not name:
            continue

        rid = _slugify_token(name)
        if not rid or rid == "all":
            continue

        aliases = _region_aliases(name)
        if isinstance(item, dict):
            extra_aliases = item.get("aliases", [])
            if isinstance(extra_aliases, list):
                aliases.update(_slugify_token(v) for v in extra_aliases if str(v).strip())

        lookup[rid] = {
            "id": rid,
            "label": name,
            "aliases": {a for a in aliases if a},
        }

    return lookup


def _infer_country_id(split_name, split_cfg, country_lookup):
    if not country_lookup:
        return None

    img_root = ""
    if isinstance(split_cfg, dict):
        img_root = str(split_cfg.get("images_root", "") or "")

    hay_parts = [
        _slugify_token(split_name),
        _slugify_token(os.path.basename(img_root)),
        _slugify_token(os.path.basename(os.path.dirname(img_root))),
    ]
    hay = "_".join([p for p in hay_parts if p])
    if not hay:
        return None

    best_country = None
    best_len = -1
    for cid, meta in country_lookup.items():
        for alias in meta.get("aliases", []):
            if not alias:
                continue
            if re.search(rf"(^|_){re.escape(alias)}(_|$)", hay):
                if len(alias) > best_len:
                    best_country = cid
                    best_len = len(alias)
    if not best_country and len(country_lookup) == 1:
        return next(iter(country_lookup.keys()))
    return best_country


def _infer_region_id(split_name, split_cfg, region_lookup):
    if not region_lookup:
        return None

    img_root = ""
    if isinstance(split_cfg, dict):
        img_root = str(split_cfg.get("images_root", "") or "")

    hay_parts = [
        _slugify_token(split_name),
        _slugify_token(os.path.basename(img_root)),
        _slugify_token(os.path.basename(os.path.dirname(img_root))),
    ]
    hay = "_".join([p for p in hay_parts if p])
    if not hay:
        return None

    best_region = None
    best_len = -1
    for rid, meta in region_lookup.items():
        for alias in meta.get("aliases", []):
            if not alias:
                continue
            if re.search(rf"(^|_){re.escape(alias)}(_|$)", hay):
                if len(alias) > best_len:
                    best_region = rid
                    best_len = len(alias)
    if not best_region and len(region_lookup) == 1:
        return next(iter(region_lookup.keys()))
    return best_region


def _merge_split_stats(scoped_splits):
    class_dist = defaultdict(int)
    resolution_hist = defaultdict(int)
    merged_raw = {
        "objects_per_image": [],
        "bbox_area_rel": [],
        "class_sets_per_image": [],
        "class_counts_per_image": [],
        "class_area_sum_per_image": [],
        "image_paths": [],
        "bbox_area_rel_by_class": defaultdict(list),
        "bbox_shapes_by_class": defaultdict(list),
    }

    total_images = 0
    total_annotations = 0
    total_images_with_annotations = 0

    for split_stats in scoped_splits.values():
        if not isinstance(split_stats, dict):
            continue
        total_images += int(split_stats.get("num_images", 0))
        total_annotations += int(split_stats.get("num_annotations", 0))
        total_images_with_annotations += int(split_stats.get("num_images_with_annotations", 0))

        for cls, val in (split_stats.get("class_distribution", {}) or {}).items():
            class_dist[cls] += int(val)

        for res_key, cnt in (split_stats.get("image_resolution_histogram", {}) or {}).items():
            resolution_hist[str(res_key)] += int(cnt)

        raw = split_stats.get("_raw", {}) or {}
        merged_raw["objects_per_image"].extend(raw.get("objects_per_image", []) or [])
        merged_raw["bbox_area_rel"].extend(raw.get("bbox_area_rel", []) or [])
        merged_raw["class_sets_per_image"].extend(raw.get("class_sets_per_image", []) or [])
        merged_raw["class_counts_per_image"].extend(raw.get("class_counts_per_image", []) or [])
        merged_raw["class_area_sum_per_image"].extend(raw.get("class_area_sum_per_image", []) or [])
        merged_raw["image_paths"].extend(raw.get("image_paths", []) or [])

        for cls, vals in (raw.get("bbox_area_rel_by_class", {}) or {}).items():
            merged_raw["bbox_area_rel_by_class"][cls].extend(vals or [])
        for cls, vals in (raw.get("bbox_shapes_by_class", {}) or {}).items():
            merged_raw["bbox_shapes_by_class"][cls].extend(vals or [])

    if not total_images_with_annotations and merged_raw["objects_per_image"]:
        total_images_with_annotations = sum(
            1 for v in merged_raw["objects_per_image"] if v and v > 0
        )

    counts = list(class_dist.values())
    if counts:
        imbalance = {
            "max_class_count": max(counts),
            "min_class_count": min(counts),
            "ratio": max(counts) / max(1, min(counts)),
        }
    else:
        imbalance = {"max_class_count": 0, "min_class_count": 0, "ratio": 0}

    return {
        "num_images": total_images,
        "num_annotations": total_annotations,
        "num_images_with_annotations": total_images_with_annotations,
        "annotation_coverage": (
            total_images_with_annotations / max(1, total_images)
        ),
        "class_distribution": dict(class_dist),
        "image_resolution_histogram": dict(
            sorted(
                resolution_hist.items(),
                key=lambda kv: (
                    int(kv[0].split("x")[0]) if "x" in kv[0] and kv[0].split("x")[0].isdigit() else 10**9,
                    int(kv[0].split("x")[1]) if "x" in kv[0] and kv[0].split("x")[1].isdigit() else 10**9,
                ),
            )
        ),
        "num_classes": len(class_dist),
        "imbalance": imbalance,
        "objects_per_image_hist": dict(
            sorted(Counter(merged_raw["objects_per_image"]).items(), key=lambda kv: kv[0])
        ),
        "class_sizes": {
            cls: _safe_stats(vals)
            for cls, vals in merged_raw["bbox_area_rel_by_class"].items()
        },
        "_raw": {
            "objects_per_image": merged_raw["objects_per_image"],
            "bbox_area_rel": merged_raw["bbox_area_rel"],
            "class_sets_per_image": merged_raw["class_sets_per_image"],
            "class_counts_per_image": merged_raw["class_counts_per_image"],
            "class_area_sum_per_image": merged_raw["class_area_sum_per_image"],
            "image_paths": merged_raw["image_paths"],
            "bbox_area_rel_by_class": dict(merged_raw["bbox_area_rel_by_class"]),
            "bbox_shapes_by_class": dict(merged_raw["bbox_shapes_by_class"]),
        },
    }


def _build_scoped_stats_data(stats_data, split_names):
    split_names = list(split_names or [])
    all_splits = stats_data.get("splits", {}) or {}
    scoped_splits = {name: all_splits[name] for name in split_names if name in all_splits}

    return {
        "dataset_name": stats_data.get("dataset_name"),
        "dataset_id": stats_data.get("dataset_id"),
        "task_type": stats_data.get("task_type"),
        "format": stats_data.get("format"),
        "computed_at": stats_data.get("computed_at"),
        "global": _merge_split_stats(scoped_splits),
        "splits": scoped_splits,
    }


def _build_view_payload(view_id, label, split_names, scoped_stats_data):
    g = scoped_stats_data.get("global", {})
    n_images = int(g.get("num_images", 0))
    n_images_with_annotations = int(g.get("num_images_with_annotations", 0))
    class_distribution = dict(g.get("class_distribution", {}) or {})
    res_hist = g.get("image_resolution_histogram", {}) or {}
    res_details = [
        {"resolution": str(k), "count": int(v)}
        for k, v in res_hist.items()
    ]
    res_list = [str(k) for k in res_hist.keys()]

    return {
        "id": view_id,
        "label": label,
        "split_names": split_names,
        "summary": {
            "num_images": n_images,
            "num_annotations": int(g.get("num_annotations", 0)),
            "num_images_with_annotations": n_images_with_annotations,
            "num_images_without_annotations": max(0, n_images - n_images_with_annotations),
            "num_classes": len(class_distribution),
            "annotation_coverage": round(float(g.get("annotation_coverage", 0.0)), 6),
            "class_distribution": class_distribution,
            "resolutions": res_list,
            "resolution_details": res_details,
        },
        "artifacts": {
            "class_balance": _build_class_balance(scoped_stats_data),
            "class_cooccurrence": _build_class_cooccurrence(scoped_stats_data),
            "class_sizes": _build_class_sizes(scoped_stats_data),
            "objects_distribution": _build_objects_distribution(scoped_stats_data),
            "classes_per_image": _build_classes_per_image(scoped_stats_data),
            "classes_treemap": _build_classes_treemap(scoped_stats_data),
        },
    }


def _build_country_views(config, stats_data):
    splits_stats = stats_data.get("splits", {}) or {}
    resolved_split_map = {s.get("name"): s for s in _resolved_splits(config)}
    country_lookup = _build_country_lookup(config)

    # Skip country drilldowns for single-country datasets.
    if country_lookup and len(country_lookup) <= 1:
        return None

    groups = defaultdict(list)
    for split_name in splits_stats.keys():
        cid = _infer_country_id(split_name, resolved_split_map.get(split_name, {}), country_lookup)
        if cid:
            groups[cid].append(split_name)

    all_split_names = sorted(list(splits_stats.keys()))
    all_stats = _build_scoped_stats_data(stats_data, all_split_names)
    views = {
        "all": _build_view_payload("all", "All", all_split_names, all_stats)
    }
    view_order = ["all"]

    for cid, meta in country_lookup.items():
        split_names = sorted(groups.get(cid, []))
        if not split_names:
            continue
        scoped = _build_scoped_stats_data(stats_data, split_names)
        views[cid] = _build_view_payload(cid, meta.get("label", cid), split_names, scoped)
        view_order.append(cid)

    return {
        "version": 1,
        "default_view": "all",
        "view_order": view_order,
        "views": views,
    }


def _build_region_views(config, stats_data):
    splits_stats = stats_data.get("splits", {}) or {}
    resolved_split_map = {s.get("name"): s for s in _resolved_splits(config)}
    region_lookup = _build_region_lookup(config)

    if not region_lookup:
        return None

    groups = defaultdict(list)
    for split_name in splits_stats.keys():
        rid = _infer_region_id(split_name, resolved_split_map.get(split_name, {}), region_lookup)
        if rid:
            groups[rid].append(split_name)

    all_split_names = sorted(list(splits_stats.keys()))
    all_stats = _build_scoped_stats_data(stats_data, all_split_names)
    views = {
        "all": _build_view_payload("all", "All", all_split_names, all_stats)
    }
    view_order = ["all"]

    for rid, meta in region_lookup.items():
        split_names = sorted(groups.get(rid, []))
        if not split_names:
            continue
        scoped = _build_scoped_stats_data(stats_data, split_names)
        views[rid] = _build_view_payload(rid, meta.get("label", rid), split_names, scoped)
        view_order.append(rid)

    if len(view_order) <= 1:
        return None

    return {
        "version": 1,
        "default_view": "all",
        "view_order": view_order,
        "views": views,
    }


def _compute_class_metrics(stats_data):
    """
    Compute per-class metrics for web tables/charts.
    """
    raw = _extract_global_raw(stats_data)
    class_counts_per_image = raw.get("class_counts_per_image", [])
    class_area_sum_per_image = raw.get("class_area_sum_per_image", [])
    classes = _sorted_classes(stats_data)

    # Keep aligned lengths for robust iteration.
    n = min(len(class_counts_per_image), len(class_area_sum_per_image))
    class_counts_per_image = class_counts_per_image[:n]
    class_area_sum_per_image = class_area_sum_per_image[:n]

    metrics = {}
    for cls in classes:
        images_with_class = 0
        objects = 0
        area_sum_on_positive_images = 0.0

        for i in range(n):
            c_map = class_counts_per_image[i] or {}
            a_map = class_area_sum_per_image[i] or {}
            c = int(c_map.get(cls, 0))
            if c > 0:
                images_with_class += 1
                objects += c
                area_sum_on_positive_images += float(a_map.get(cls, 0.0))

        avg_count_on_image = (objects / images_with_class) if images_with_class else 0.0
        avg_area_on_image = (area_sum_on_positive_images / images_with_class) if images_with_class else 0.0

        metrics[cls] = {
            "images": images_with_class,
            "objects": objects,
            "count_on_image_avg": avg_count_on_image,
            "area_on_image_avg": avg_area_on_image,
        }

    return metrics


def _build_class_balance(stats_data):
    """
    Detailed class balance table for web:
    - Class
    - Images
    - Objects
    - Count on image, average
    - Area on image, average
    """
    metrics = _compute_class_metrics(stats_data)
    rows = []

    for cls in _sorted_classes(stats_data):
        m = metrics.get(cls, {})
        rows.append(
            [
                cls,
                int(m.get("images", 0)),
                int(m.get("objects", 0)),
                round(float(m.get("count_on_image_avg", 0.0)), 4),
                round(float(m.get("area_on_image_avg", 0.0)) * 100.0, 4),
            ]
        )

    return {
        "columns": [
            "Class",
            "Images",
            "Objects",
            "Count on image, average",
            "Area on image, average (%)",
        ],
        "data": rows,
        "description": "Images column counts images containing at least one object of class.",
        "options": {"sort": {"columnIndex": 2, "order": "desc"}, "pageSize": 100},
    }


def _build_class_cooccurrence(stats_data):
    class_dist = stats_data.get("global", {}).get("class_distribution", {})
    classes = sorted(class_dist.keys())
    idx = {c: i for i, c in enumerate(classes)}
    matrix = [[0 for _ in classes] for _ in classes]

    class_sets = _extract_global_raw(stats_data).get("class_sets_per_image", [])
    for s in class_sets:
        unique = sorted(set(s or []))
        for i in range(len(unique)):
            for j in range(i, len(unique)):
                a, b = unique[i], unique[j]
                if a in idx and b in idx:
                    matrix[idx[a]][idx[b]] += 1
                    if a != b:
                        matrix[idx[b]][idx[a]] += 1

    return {
        "type": "matrix",
        "labels": classes,
        "data": matrix,
        "title": "Co-occurrence Matrix",
        "description": "Cell [i,j] is number of images that contain both classes i and j.",
    }


def _build_objects_distribution(stats_data):
    """
    Class-wise objects-per-image distribution heatmap.
    For each class, x-axis is object count on image (0..max), value is number of images.
    """
    raw = _extract_global_raw(stats_data)
    classes = _sorted_classes(stats_data)
    class_counts_per_image = raw.get("class_counts_per_image", [])

    if not classes:
        return {
            "type": "heatmap",
            "title": "Objects on images - distribution for every class",
            "x_labels": [],
            "y_labels": [],
            "z": [],
        }

    per_class_hist = {c: Counter() for c in classes}
    max_count = 0

    for cmap in class_counts_per_image:
        cmap = cmap or {}
        for cls in classes:
            c = int(cmap.get(cls, 0))
            per_class_hist[cls][c] += 1
            if c > max_count:
                max_count = c

    x_labels = [str(i) for i in range(max_count + 1)]
    z = []
    for cls in classes:
        row = [per_class_hist[cls].get(i, 0) for i in range(max_count + 1)]
        z.append(row)

    return {
        "type": "heatmap",
        "title": "Objects on images - distribution for every class",
        "x_labels": x_labels,
        "y_labels": classes,
        "z": z,
        "description": "Value = number of images where class has exactly x objects.",
    }


def _build_classes_per_image(stats_data):
    class_sets = _extract_global_raw(stats_data).get("class_sets_per_image", [])
    counts = [len(set(s or [])) for s in class_sets]
    hist = Counter(counts)
    max_x = max(hist.keys()) if hist else 0
    points = [{"x": str(i), "y": hist.get(i, 0)} for i in range(max_x + 1)]

    return {
        "type": "histogram",
        "title": "Distinct classes per image",
        "series": [{"name": "classes", "data": points}],
    }


def _build_class_sizes(stats_data):
    """
    Table with class size properties similar to datasetninja presentation.
    """
    by_class = _extract_global_raw(stats_data).get("bbox_shapes_by_class", {})
    rows = []

    for cls in _sorted_classes(stats_data):
        shapes = by_class.get(cls, [])
        if not shapes:
            rows.append([cls, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            continue

        areas = [float(s.get("area_rel", 0.0)) for s in shapes]
        h_rel = [float(s.get("height_rel", 0.0)) for s in shapes]
        h_px = [float(s.get("height_px", 0.0)) for s in shapes]

        rows.append(
            [
                cls,
                len(shapes),
                round((sum(areas) / len(areas)) * 100.0, 4),
                round(max(areas) * 100.0, 4),
                round(min(areas) * 100.0, 4),
                round(min(h_px), 2),
                round(min(h_rel) * 100.0, 4),
                round(max(h_px), 2),
                round(max(h_rel) * 100.0, 4),
                round(sum(h_px) / len(h_px), 2),
                round((sum(h_rel) / len(h_rel)) * 100.0, 4),
            ]
        )

    rows.sort(key=lambda r: r[1], reverse=True)

    return {
        "columns": [
            "Class",
            "Object count",
            "Avg area (%)",
            "Max area (%)",
            "Min area (%)",
            "Min height (px)",
            "Min height (%)",
            "Max height (px)",
            "Max height (%)",
            "Avg height (px)",
            "Avg height (%)",
        ],
        "data": rows,
        "description": "Area and height are relative to image size where (%) columns are 0-100 scale.",
    }


def _build_classes_treemap(stats_data):
    """
    Treemap uses average class area on image (positive images only), matching requested view.
    """
    metrics = _compute_class_metrics(stats_data)
    data = []
    for cls in _sorted_classes(stats_data):
        avg_area_pct = float(metrics.get(cls, {}).get("area_on_image_avg", 0.0)) * 100.0
        data.append({"x": cls, "y": round(avg_area_pct, 4)})

    return {
        "type": "treemap",
        "title": "Average area of class objects on image",
        "series": [{"data": data}],
        "value_unit": "%",
    }


def _safe_filename(name):
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    return safe.strip("_") or "class"


def _class_color(name):
    seed = abs(hash(str(name)))
    return (
        50 + (seed % 180),
        50 + ((seed // 17) % 180),
        50 + ((seed // 37) % 180),
    )


def _resolve_path(path_value, config_dir):
    if not path_value:
        return ""
    if os.path.isabs(path_value):
        return os.path.normpath(path_value)
    return os.path.normpath(os.path.join(config_dir, path_value))


def _resolved_splits(config):
    cfg_dir = config.get("_config_dir", os.getcwd())
    global_class_map = dict(config.get("class_map", {}) or {})
    global_class_exclude = list(config.get("class_exclude", []) or [])
    splits = config.get("splits")
    if not splits and config.get("data"):
        splits = {"data": config.get("data")}
    splits = splits or {}

    out = []
    for name, scfg in splits.items():
        scfg = scfg or {}
        split_class_map = scfg.get("class_map")
        if split_class_map is None:
            split_class_map = global_class_map
        split_class_exclude = scfg.get("class_exclude")
        if split_class_exclude is None:
            split_class_exclude = global_class_exclude

        out.append(
            {
                "name": name,
                "images_root": _resolve_path(scfg.get("images_root", ""), cfg_dir),
                "annotations": _resolve_path(scfg.get("annotations", ""), cfg_dir),
                "masks_root": _resolve_path(scfg.get("masks_root", ""), cfg_dir),
                "mask_suffixes": list(scfg.get("mask_suffixes", [""]) or [""]),
                "class_map": dict(split_class_map or {}),
                "class_exclude": list(split_class_exclude or []),
            }
        )
    return out


def _match_split_for_image(image_path, splits):
    img = os.path.normcase(os.path.normpath(image_path))
    best = None
    best_len = -1
    for s in splits:
        root = s.get("images_root", "")
        if not root:
            continue
        root_n = os.path.normcase(os.path.normpath(root))
        if img == root_n or img.startswith(root_n + os.sep):
            if len(root_n) > best_len:
                best = s
                best_len = len(root_n)
    return best


def _load_coco_cache(coco_json_path, cache):
    key = os.path.normcase(os.path.normpath(coco_json_path))
    if key in cache:
        return cache[key]

    if not coco_json_path or not os.path.isfile(coco_json_path):
        cache[key] = None
        return None

    try:
        with open(coco_json_path, "r", encoding="utf-8-sig") as f:
            payload = json.load(f)
    except Exception:
        cache[key] = None
        return None

    images = payload.get("images", [])
    anns = payload.get("annotations", [])
    categories = {c.get("id"): c.get("name", str(c.get("id"))) for c in payload.get("categories", [])}
    ann_by_image = {}
    for ann in anns:
        ann_by_image.setdefault(ann.get("image_id"), []).append(ann)

    by_file = {}
    for img in images:
        fname = img.get("file_name", "")
        base = os.path.basename(fname)
        key_name = base.lower()
        by_file[key_name] = {"image": img, "annotations": ann_by_image.get(img.get("id"), [])}

    cache[key] = {"categories": categories, "by_file": by_file}
    return cache[key]


def _find_sample_annotation_info(sample_src, config, splits, coco_cache):
    fmt = str(config.get("format", "")).strip().lower()
    split = _match_split_for_image(sample_src, splits)
    basename = os.path.splitext(os.path.basename(sample_src))[0]
    sample_key = os.path.basename(sample_src).lower()

    if fmt == "voc" and split:
        xml_path = os.path.join(split.get("annotations", ""), basename + ".xml")
        if os.path.isfile(xml_path):
            return {"mode": "bbox", "format": "voc", "source_path": xml_path}

    if fmt == "yolo" and split:
        txt_path = os.path.join(split.get("annotations", ""), basename + ".txt")
        if os.path.isfile(txt_path):
            return {
                "mode": "bbox",
                "format": "yolo",
                "source_path": txt_path,
                "class_map": dict(split.get("class_map", {}) or {}),
                "class_exclude": list(split.get("class_exclude", []) or []),
            }

    if fmt == "coco" and split:
        coco_path = split.get("annotations", "")
        coco = _load_coco_cache(coco_path, coco_cache)
        if coco:
            rec = coco["by_file"].get(sample_key)
            if not rec:
                # Fallback by basename match without extension.
                for k, v in coco["by_file"].items():
                    if os.path.splitext(k)[0] == basename.lower():
                        rec = v
                        break
            if rec:
                return {
                    "mode": "bbox",
                    "format": "coco",
                    "source_path": coco_path,
                    "coco_record": {
                        "image": rec.get("image", {}),
                        "annotations": rec.get("annotations", []),
                        "categories": coco.get("categories", {}),
                    },
                }

    if fmt == "png_masks" and split:
        mroot = split.get("masks_root", "")
        suffixes = split.get("mask_suffixes", [""]) if isinstance(split, dict) else [""]
        if not isinstance(suffixes, (list, tuple)):
            suffixes = [""]
        suffixes = [str(s) for s in suffixes if s is not None]
        if "" not in suffixes:
            suffixes.append("")
        for suffix in suffixes:
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
                mp = os.path.join(mroot, f"{basename}{suffix}{ext}")
                if os.path.isfile(mp):
                    return {
                        "mode": "mask",
                        "format": "png_masks",
                        "source_path": mp,
                        "class_map": dict(split.get("class_map", {}) or {}),
                        "class_exclude": list(split.get("class_exclude", []) or []),
                        "min_component_area_px": split.get("min_component_area_px", 1),
                    }

    return None


def _parse_voc_boxes(xml_path):
    out = []
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return out
    for obj in root.findall("object"):
        cls = obj.findtext("name", default="object")
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            xmin = float(bb.findtext("xmin", default="0"))
            ymin = float(bb.findtext("ymin", default="0"))
            xmax = float(bb.findtext("xmax", default="0"))
            ymax = float(bb.findtext("ymax", default="0"))
        except Exception:
            continue
        out.append({"class": cls, "bbox": [xmin, ymin, xmax, ymax]})
    return out


def _parse_yolo_boxes(txt_path, iw, ih):
    out = []
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return out

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = parts[0]
        try:
            cx = float(parts[1]) * iw
            cy = float(parts[2]) * ih
            bw = float(parts[3]) * iw
            bh = float(parts[4]) * ih
        except Exception:
            continue
        xmin = cx - bw / 2.0
        ymin = cy - bh / 2.0
        xmax = cx + bw / 2.0
        ymax = cy + bh / 2.0
        out.append({"class": cls, "bbox": [xmin, ymin, xmax, ymax]})
    return out


def _parse_coco_boxes(coco_record):
    out = []
    cats = coco_record.get("categories", {})
    for ann in coco_record.get("annotations", []):
        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            continue
        x, y, w, h = bbox
        cls_id = ann.get("category_id")
        cls = cats.get(cls_id, str(cls_id))
        out.append({"class": cls, "bbox": [x, y, x + w, y + h]})
    return out


def _get_voc_canvas_size(xml_path):
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None

    size_node = root.find("size")
    if size_node is None:
        return None
    try:
        w = int(float(size_node.findtext("width", default="0")))
        h = int(float(size_node.findtext("height", default="0")))
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return (w, h)


def _get_mask_canvas_size(mask_path):
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        with Image.open(mask_path) as m:
            w, h = m.size
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return (int(w), int(h))


def _get_annotation_canvas_size(ann_info):
    if not isinstance(ann_info, dict):
        return None

    ann_format = str(ann_info.get("format", "")).strip().lower()
    if ann_format == "voc":
        return _get_voc_canvas_size(ann_info.get("source_path", ""))

    if ann_format == "coco":
        image_meta = (ann_info.get("coco_record") or {}).get("image") or {}
        try:
            w = int(float(image_meta.get("width", 0)))
            h = int(float(image_meta.get("height", 0)))
        except Exception:
            return None
        if w > 0 and h > 0:
            return (w, h)
        return None

    if ann_format == "png_masks":
        return _get_mask_canvas_size(ann_info.get("source_path", ""))

    return None


def _resize_sample_image(image_path, max_side=1280, target_canvas_size=None):
    """
    Resize sample image in-place while preserving aspect ratio.
    If target_canvas_size is provided, prefer EXIF-normalized orientation when it matches
    annotation canvas dimensions, so overlay coordinates stay in the same frame.
    """
    try:
        from PIL import Image, ImageOps
    except Exception:
        return None

    try:
        with Image.open(image_path) as src_img:
            raw_w, raw_h = src_img.size
            exif_img = ImageOps.exif_transpose(src_img)
            exif_w, exif_h = exif_img.size

            use_exif_frame = False
            if (
                isinstance(target_canvas_size, (list, tuple))
                and len(target_canvas_size) == 2
            ):
                try:
                    target_w = int(target_canvas_size[0])
                    target_h = int(target_canvas_size[1])
                except Exception:
                    target_w, target_h = 0, 0

                if target_w > 0 and target_h > 0:
                    raw_match = (raw_w == target_w and raw_h == target_h)
                    exif_match = (exif_w == target_w and exif_h == target_h)
                    if exif_match and not raw_match:
                        use_exif_frame = True
                    elif not raw_match and not exif_match:
                        raw_delta = abs(raw_w - target_w) + abs(raw_h - target_h)
                        exif_delta = abs(exif_w - target_w) + abs(exif_h - target_h)
                        use_exif_frame = exif_delta < raw_delta

            img = exif_img if use_exif_frame else src_img.copy()
            img = img.convert("RGB")
            old_w, old_h = img.size
            if old_w <= 0 or old_h <= 0:
                return None

            if not max_side or max(old_w, old_h) <= int(max_side):
                new_w, new_h = old_w, old_h
                resized = img
            else:
                scale = float(max_side) / float(max(old_w, old_h))
                new_w = max(1, int(round(old_w * scale)))
                new_h = max(1, int(round(old_h * scale)))
                resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
                resized = img.resize((new_w, new_h), resample=resample)

            resized.save(image_path)
    except Exception:
        return None

    sx = float(new_w) / float(old_w)
    sy = float(new_h) / float(old_h)
    return {
        "old_width": old_w,
        "old_height": old_h,
        "new_width": new_w,
        "new_height": new_h,
        "scale_x": sx,
        "scale_y": sy,
        "used_exif_orientation": use_exif_frame,
    }


def _scale_voc_annotation(src_xml, dst_xml, sx, sy, new_w, new_h):
    try:
        root = ET.parse(src_xml).getroot()
    except Exception:
        return False

    size_node = root.find("size")
    if size_node is not None:
        w_node = size_node.find("width")
        h_node = size_node.find("height")
        if w_node is not None:
            w_node.text = str(int(new_w))
        if h_node is not None:
            h_node.text = str(int(new_h))

    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        for key, scale, limit in [
            ("xmin", sx, max(0, int(new_w) - 1)),
            ("xmax", sx, max(0, int(new_w) - 1)),
            ("ymin", sy, max(0, int(new_h) - 1)),
            ("ymax", sy, max(0, int(new_h) - 1)),
        ]:
            node = bb.find(key)
            if node is None or node.text is None:
                continue
            try:
                v = int(round(float(node.text) * scale))
            except Exception:
                continue
            v = max(0, min(limit, v))
            node.text = str(v)

    try:
        ET.ElementTree(root).write(dst_xml, encoding="utf-8", xml_declaration=True)
        return True
    except Exception:
        return False


def _scale_coco_record(coco_record, sx, sy, new_w, new_h):
    scaled = copy.deepcopy(coco_record or {})
    img = scaled.get("image", {}) if isinstance(scaled, dict) else {}
    if isinstance(img, dict):
        img["width"] = int(new_w)
        img["height"] = int(new_h)

    anns = scaled.get("annotations", []) if isinstance(scaled, dict) else []
    for ann in anns:
        bbox = ann.get("bbox", [])
        if isinstance(bbox, list) and len(bbox) == 4:
            ann["bbox"] = [
                float(bbox[0]) * sx,
                float(bbox[1]) * sy,
                float(bbox[2]) * sx,
                float(bbox[3]) * sy,
            ]
        if "area" in ann:
            try:
                ann["area"] = float(ann.get("area", 0.0)) * sx * sy
            except Exception:
                pass

        seg = ann.get("segmentation")
        if isinstance(seg, list):
            new_seg = []
            for poly in seg:
                if not isinstance(poly, list):
                    new_seg.append(poly)
                    continue
                out = []
                for i, val in enumerate(poly):
                    try:
                        fv = float(val)
                    except Exception:
                        fv = 0.0
                    out.append(fv * (sx if i % 2 == 0 else sy))
                new_seg.append(out)
            ann["segmentation"] = new_seg

    return scaled


def _copy_or_scale_yolo_annotation(src_txt, dst_txt):
    try:
        shutil.copy2(src_txt, dst_txt)
        return True
    except Exception:
        return False


def _mask_to_polygons(
    mask_path,
    out_json_path,
    target_size,
    class_map=None,
    class_exclude=None,
    min_component_area_px=1,
):
    """Convert indexed mask classes to pixel-precise polygon JSON without OpenCV dependency."""
    try:
        import numpy as np
    except Exception:
        return False

    try:
        from PIL import Image
        with Image.open(mask_path) as mask_img:
            mask_img = mask_img.convert("L")
            mask_arr = np.array(mask_img)
    except Exception:
        return False

    if mask_arr.ndim != 2:
        return False

    src_h, src_w = mask_arr.shape
    if src_h <= 0 or src_w <= 0:
        return False

    out_w, out_h = src_w, src_h
    if isinstance(target_size, (list, tuple)) and len(target_size) == 2:
        try:
            tw = int(target_size[0])
            th = int(target_size[1])
        except Exception:
            tw, th = 0, 0
        if tw > 0 and th > 0:
            out_w, out_h = tw, th

    def _parse_int_mask_value(value):
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return int(value)
        s = str(value or "").strip()
        if not s:
            return None
        if s.startswith("+"):
            s = s[1:]
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            try:
                return int(s)
            except Exception:
                return None
        return None

    excluded = {str(v).strip() for v in (class_exclude or []) if str(v).strip()}
    try:
        min_component_area_px = int(min_component_area_px)
    except Exception:
        min_component_area_px = 1
    if min_component_area_px < 1:
        min_component_area_px = 1

    def _infer_foreground_values(mask_array, cmap, cexclude):
        vals, counts = np.unique(mask_array, return_counts=True)
        if vals.size == 0:
            return set()
        if vals.size == 1:
            return set()

        present_vals = {int(v) for v in vals.tolist()}
        mapped_vals = set()
        excluded_raw = set()
        if isinstance(cmap, dict):
            for raw_key, mapped_name in cmap.items():
                parsed = _parse_int_mask_value(raw_key)
                raw_key_str = str(raw_key).strip()
                mapped_str = str(mapped_name).strip()
                if raw_key_str in cexclude or mapped_str in cexclude:
                    if parsed is not None:
                        excluded_raw.add(parsed)
                    continue
                if parsed is not None and parsed in present_vals and str(parsed) not in cexclude:
                    mapped_vals.add(parsed)

        # Binary masks: cracks are usually minority pixels. This also handles inverse masks.
        if vals.size == 2:
            idx = int(np.argmin(counts))
            minority_val = int(vals[idx])
            minority_ratio = float(counts[idx]) / max(1.0, float(np.sum(counts)))
            labels = set()
            if isinstance(cmap, dict):
                labels = {str(v).strip() for v in cmap.values() if str(v).strip()}
            if (not cmap or len(labels) <= 1) and minority_ratio <= 0.49:
                return {minority_val}

        if mapped_vals:
            return mapped_vals

        non_zero = {
            v
            for v in present_vals
            if v != 0 and v not in excluded_raw and str(v) not in cexclude
        }
        if non_zero:
            return non_zero

        return set()

    binary_values = _infer_foreground_values(mask_arr, class_map, excluded)
    if not binary_values:
        payload = {
            "format": "polygon",
            "source": "mask",
            "image": {
                "width": int(out_w),
                "height": int(out_h),
            },
            "polygons": [],
        }
        try:
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            return True
        except Exception:
            return False

    def _dilate_once(mask):
        h_, w_ = mask.shape
        out = mask.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                y_src0 = max(0, -dy)
                y_src1 = h_ - max(0, dy)
                x_src0 = max(0, -dx)
                x_src1 = w_ - max(0, dx)
                if y_src1 <= y_src0 or x_src1 <= x_src0:
                    continue
                y_dst0 = max(0, dy)
                y_dst1 = y_dst0 + (y_src1 - y_src0)
                x_dst0 = max(0, dx)
                x_dst1 = x_dst0 + (x_src1 - x_src0)
                out[y_dst0:y_dst1, x_dst0:x_dst1] = np.logical_or(
                    out[y_dst0:y_dst1, x_dst0:x_dst1],
                    mask[y_src0:y_src1, x_src0:x_src1],
                )
        return out

    def _polygon_area(points):
        if len(points) < 3:
            return 0.0
        area = 0.0
        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0

    def _simplify_collinear(points):
        if len(points) < 4:
            return points
        out = []
        n = len(points)
        for i in range(n):
            x0, y0 = points[(i - 1) % n]
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            # Keep non-collinear corners only.
            if (x1 - x0) * (y2 - y1) != (y1 - y0) * (x2 - x1):
                out.append((x1, y1))
        return out if len(out) >= 3 else points

    def _trace_component_boundaries(component_pixels):
        # Build directed boundary edges around occupied pixel cells.
        pixel_set = set(component_pixels)
        edges = []
        for px, py in pixel_set:
            if (px, py - 1) not in pixel_set:
                edges.append(((px, py), (px + 1, py)))
            if (px + 1, py) not in pixel_set:
                edges.append(((px + 1, py), (px + 1, py + 1)))
            if (px, py + 1) not in pixel_set:
                edges.append(((px + 1, py + 1), (px, py + 1)))
            if (px - 1, py) not in pixel_set:
                edges.append(((px, py + 1), (px, py)))

        if not edges:
            return []

        outgoing = defaultdict(list)
        remaining = defaultdict(int)
        for s, e in edges:
            outgoing[s].append(e)
            remaining[(s, e)] += 1

        loops = []
        # Trace all closed loops and keep the largest one as the outer boundary.
        for s, e in edges:
            while remaining[(s, e)] > 0:
                start = s
                prev = s
                cur = e
                remaining[(s, e)] -= 1
                loop = [start, cur]

                guard = 0
                max_steps = max(16, len(edges) * 2)
                while cur != start and guard < max_steps:
                    candidates = [nxt for nxt in outgoing.get(cur, []) if remaining[(cur, nxt)] > 0]
                    if not candidates:
                        break
                    if len(candidates) > 1:
                        non_back = [nxt for nxt in candidates if nxt != prev]
                        candidates = non_back or candidates
                        candidates = sorted(candidates)
                    nxt = candidates[0]
                    remaining[(cur, nxt)] -= 1
                    prev, cur = cur, nxt
                    loop.append(cur)
                    guard += 1

                if cur == start and len(loop) >= 4:
                    if loop[-1] == loop[0]:
                        loop = loop[:-1]
                    loop = _simplify_collinear(loop)
                    if len(loop) >= 3:
                        loops.append(loop)

        if not loops:
            return []

        loops.sort(key=_polygon_area, reverse=True)
        return loops

    def _class_value_label_pairs(values, cmap, cexclude):
        values_sorted = sorted(int(v) for v in values)
        value_to_label = {}

        if isinstance(cmap, dict):
            for raw_key, mapped_name in cmap.items():
                parsed = _parse_int_mask_value(raw_key)
                if parsed is None or parsed not in values:
                    continue
                raw_key_str = str(raw_key).strip()
                mapped_str = str(mapped_name).strip()
                if (
                    raw_key_str in cexclude
                    or str(parsed) in cexclude
                    or mapped_str in cexclude
                ):
                    continue
                if mapped_str and parsed not in value_to_label:
                    value_to_label[parsed] = mapped_str

        pairs = []
        for value in values_sorted:
            label = value_to_label.get(value)
            if not label:
                if len(values_sorted) == 1:
                    if isinstance(cmap, dict) and cmap:
                        labels = sorted(
                            {
                                str(v).strip()
                                for v in cmap.values()
                                if str(v).strip() and str(v).strip() not in cexclude
                            }
                        )
                        if len(labels) == 1:
                            label = labels[0]
                if not label:
                    label = "crack" if len(values_sorted) == 1 else f"class_{value}"
            pairs.append((value, label))
        return pairs

    def _extract_polygons_for_binary(binary_mask, label, start_pid):
        binary_local = binary_mask

        # Slightly thicken sparse masks to avoid dot-like polygon artifacts for hairline cracks.
        fill_ratio = float(np.count_nonzero(binary_local)) / max(1.0, float(binary_local.size))
        if 0.0 < fill_ratio < 0.2:
            binary_local = _dilate_once(binary_local)

        # Build preview polygons on resized masks to keep runtime bounded for
        # very high-resolution sources while matching the rendered sample canvas.
        if out_w > 0 and out_h > 0 and (src_w != out_w or src_h != out_h):
            try:
                resized_mask = Image.fromarray((binary_local.astype(np.uint8) * 255), mode="L")
                resample = getattr(getattr(Image, "Resampling", Image), "NEAREST", Image.NEAREST)
                resized_mask = resized_mask.resize((out_w, out_h), resample=resample)
                binary_local = np.array(resized_mask) > 0
            except Exception:
                pass

        h, w = binary_local.shape
        visited = np.zeros_like(binary_local, dtype=bool)

        def _scale_point(px, py):
            src_x = int(max(0, min(w - 1, px)))
            src_y = int(max(0, min(h - 1, py)))
            if out_w == w and out_h == h:
                return [src_x, src_y]
            sx = float(max(0, out_w - 1)) / float(max(1, w - 1))
            sy = float(max(0, out_h - 1)) / float(max(1, h - 1))
            x = int(round(src_x * sx))
            y = int(round(src_y * sy))
            x = int(max(0, min(out_w - 1, x)))
            y = int(max(0, min(out_h - 1, y)))
            return [x, y]

        def _normalize_ring(points):
            scaled = [_scale_point(px, py) for px, py in points]
            if len(scaled) < 3:
                return None

            compact = []
            for pt in scaled:
                if not compact or compact[-1] != pt:
                    compact.append(pt)
            if len(compact) >= 2 and compact[0] == compact[-1]:
                compact = compact[:-1]

            if len({(p[0], p[1]) for p in compact}) < 3:
                return None
            return compact

        polygons_local = []
        pid = start_pid
        ys, xs = np.where(binary_local)
        for y0, x0 in zip(ys.tolist(), xs.tolist()):
            if visited[y0, x0]:
                continue

            stack = [(y0, x0)]
            visited[y0, x0] = True
            component = []
            min_x = max_x = x0
            min_y = max_y = y0

            while stack:
                cy, cx = stack.pop()
                component.append((cx, cy))
                if cx < min_x:
                    min_x = cx
                if cx > max_x:
                    max_x = cx
                if cy < min_y:
                    min_y = cy
                if cy > max_y:
                    max_y = cy

                for ny, nx in (
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                    (cy - 1, cx - 1),
                    (cy - 1, cx + 1),
                    (cy + 1, cx - 1),
                    (cy + 1, cx + 1),
                ):
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if not binary_local[ny, nx] or visited[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))

            if len(component) < 1:
                continue
            if len(component) < min_component_area_px:
                continue

            boundaries = _trace_component_boundaries(component)
            if not boundaries:
                boundaries = [
                    [
                        (min_x, min_y),
                        (max_x + 1, min_y),
                        (max_x + 1, max_y + 1),
                        (min_x, max_y + 1),
                    ]
                ]

            outer = _normalize_ring(boundaries[0])
            if not outer:
                continue

            holes = []
            for ring in boundaries[1:]:
                hole = _normalize_ring(ring)
                if hole:
                    holes.append(hole)

            xs_outer = [p[0] for p in outer]
            ys_outer = [p[1] for p in outer]
            min_px, max_px = min(xs_outer), max(xs_outer)
            min_py, max_py = min(ys_outer), max(ys_outer)
            bbox_w = max(1, int(max_px - min_px + 1))
            bbox_h = max(1, int(max_py - min_py + 1))

            outer_area = _polygon_area([(float(x), float(y)) for x, y in outer])
            holes_area = 0.0
            for hole in holes:
                holes_area += _polygon_area([(float(x), float(y)) for x, y in hole])
            poly_area = outer_area - holes_area
            if poly_area <= 0:
                scale_area = (float(out_w) * float(out_h)) / max(1.0, float(w) * float(h))
                poly_area = float(len(component)) * scale_area

            rec = {
                "id": pid,
                "label": label,
                "area_px": float(poly_area),
                "bbox": [int(min_px), int(min_py), bbox_w, bbox_h],
                "points": outer,
            }
            if holes:
                rec["holes"] = holes
            polygons_local.append(rec)
            pid += 1

        return polygons_local, pid

    polygons = []
    pid = 1
    for class_value, class_label in _class_value_label_pairs(binary_values, class_map, excluded):
        class_binary = mask_arr == int(class_value)
        if not np.any(class_binary):
            continue
        class_polygons, pid = _extract_polygons_for_binary(class_binary, class_label, pid)
        polygons.extend(class_polygons)

    payload = {
        "format": "polygon",
        "source": "mask",
        "image": {
            "width": int(out_w),
            "height": int(out_h),
        },
        "polygons": polygons,
    }
    try:
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return True
    except Exception:
        return False


def _frame_group_key(path):
    base = os.path.splitext(os.path.basename(path))[0].lower()
    base = re.sub(r"\d+$", "", base).rstrip("_-")
    stream = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
    return f"{stream.lower()}::{base}"


def _stream_key(path):
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path)))).lower()


def _frame_index(path):
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(\d+)$", base)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _quick_select_samples(stats_data, count=10, seed=42, require_annotations=False):
    """
    Lightweight deterministic selector for large scoped views.
    Prefers annotated and multi-class images while keeping runtime low.
    """
    raw = _extract_global_raw(stats_data)
    image_paths = raw.get("image_paths", [])
    class_sets = raw.get("class_sets_per_image", [])
    class_counts = raw.get("class_counts_per_image", [])

    n = min(len(image_paths), len(class_sets), len(class_counts))
    if n <= 0:
        return []

    rng = random.Random(seed)
    candidates = []
    for i in range(n):
        p = image_paths[i]
        if not p or not os.path.isfile(p):
            continue
        classes = sorted(set(class_sets[i] or []))
        ccounts = class_counts[i] or {}
        obj_count = int(sum(ccounts.values())) if isinstance(ccounts, dict) else 0
        candidates.append(
            {
                "path": p,
                "classes": classes,
                "class_count": len(classes),
                "objects": obj_count,
                "frame_index": _frame_index(p),
            }
        )

    if not candidates:
        return []

    if require_annotations:
        candidates = [c for c in candidates if int(c.get("objects", 0)) > 0]
        if not candidates:
            return []

    desired_target = max(1, min(int(count), len(candidates)))

    # Prefer annotated images for preview samples when available.
    if not require_annotations:
        annotated_candidates = [c for c in candidates if int(c.get("objects", 0)) > 0]
        if len(annotated_candidates) >= desired_target:
            candidates = annotated_candidates

    rng.shuffle(candidates)
    candidates.sort(
        key=lambda r: (
            1 if r.get("objects", 0) > 0 else 0,
            r.get("class_count", 0),
            r.get("objects", 0),
        ),
        reverse=True,
    )

    target = max(1, min(int(count), len(candidates)))

    # Round-robin by source stream first, then avoid near-duplicate sequential frames.
    streams = defaultdict(list)
    for rec in candidates:
        streams[_stream_key(rec["path"])].append(rec)

    stream_order = list(streams.keys())
    rng.shuffle(stream_order)

    selected = []
    seen_paths = set()
    seen_groups = set()
    selected_frame_idx = defaultdict(list)

    # Try to keep selected frames well separated on each source stream.
    min_frame_gap = 150

    def _frame_gap(stream, idx):
        if idx is None:
            return None
        prev = selected_frame_idx.get(stream, [])
        if not prev:
            return None
        return min(abs(idx - p) for p in prev)

    def _pick_from_bucket(stream, bucket):
        # Pass 1: enforce min frame gap where frame indices are available.
        gap_pool = []
        # Pass 2: fallback to best available gap if strict constraint cannot be satisfied.
        relaxed_pool = []

        for i, rec in enumerate(bucket):
            p = rec["path"]
            if p in seen_paths:
                continue

            idx = rec.get("frame_index")
            gap = _frame_gap(stream, idx)
            gap_score = gap if gap is not None else 10**9
            score = (
                gap_score,
                rec.get("class_count", 0),
                rec.get("objects", 0),
            )
            relaxed_pool.append((score, i, rec))

            if gap is None or gap >= min_frame_gap:
                gap_pool.append((score, i, rec))

        if gap_pool:
            gap_pool.sort(key=lambda t: t[0], reverse=True)
            return gap_pool[0][1]

        if relaxed_pool:
            relaxed_pool.sort(key=lambda t: t[0], reverse=True)
            return relaxed_pool[0][1]

        return None

    made_progress = True
    while len(selected) < target and made_progress:
        made_progress = False
        for s in stream_order:
            if len(selected) >= target:
                break
            bucket = streams.get(s, [])
            pick_idx = _pick_from_bucket(s, bucket)
            if pick_idx is None:
                continue

            rec = bucket.pop(pick_idx)
            p = rec["path"]
            selected.append(rec)
            seen_paths.add(p)
            seen_groups.add(_frame_group_key(p))
            fidx = rec.get("frame_index")
            if fidx is not None:
                selected_frame_idx[s].append(fidx)
            made_progress = True

    if len(selected) < target:
        for rec in candidates:
            p = rec["path"]
            if p in seen_paths:
                continue
            selected.append(rec)
            seen_paths.add(p)
            if len(selected) >= target:
                break

    return selected[:target]


def _build_spatial_heatmaps(stats_data, output_dir, relative_dir="visualizations/spatial_heatmaps", grid_size=64):
    """
    Generate per-class spatial heatmap images based on normalized bbox centers.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Warning: matplotlib not available, skipping spatial heatmaps.")
        return {"generated": False, "classes": []}

    rel_dir = str(relative_dir or "visualizations/spatial_heatmaps").replace("\\", "/").strip("/")
    by_class = _extract_global_raw(stats_data).get("bbox_shapes_by_class", {})
    vis_dir = os.path.join(output_dir, *rel_dir.split("/"))
    _safe_makedirs(vis_dir)

    heatmap_index = []
    for cls in _sorted_classes(stats_data):
        shapes = by_class.get(cls, [])
        if not shapes:
            continue

        grid = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]
        for s in shapes:
            cx = float(s.get("cx_rel", 0.5))
            cy = float(s.get("cy_rel", 0.5))
            w_rel = max(0.0, float(s.get("width_rel", 0.0)))
            h_rel = max(0.0, float(s.get("height_rel", 0.0)))

            x = min(grid_size - 1, max(0, int(cx * grid_size)))
            y = min(grid_size - 1, max(0, int(cy * grid_size)))

            # Small Gaussian-like kernel radius proportional to bbox size.
            radius = max(1, int(max(w_rel, h_rel) * grid_size * 0.6))
            for yy in range(max(0, y - radius), min(grid_size, y + radius + 1)):
                for xx in range(max(0, x - radius), min(grid_size, x + radius + 1)):
                    dx = xx - x
                    dy = yy - y
                    dist2 = dx * dx + dy * dy
                    sigma2 = max(1.0, (radius * 0.7) ** 2)
                    weight = math.exp(-dist2 / (2.0 * sigma2))
                    grid[yy][xx] += weight

        fname = f"{_safe_filename(cls)}.png"
        fpath = os.path.join(vis_dir, fname)

        vmax = max(max(row) for row in grid) if grid else 0.0
        if vmax > 0:
            norm_grid = [[v / vmax for v in row] for row in grid]
        else:
            norm_grid = grid

        # Save only the heatmap raster (no title/axes/colorbar/background).
        plt.imsave(fpath, norm_grid, cmap="viridis")

        heatmap_index.append({"class": cls, "file": f"{rel_dir}/{fname}"})

    return {"generated": True, "dir": rel_dir, "classes": heatmap_index}


def _plot_pngs(stats_dir, artifacts):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Warning: matplotlib not available, skipping PNG plot artifacts.")
        return

    # class_balance.png (objects per class)
    cb = artifacts["class_balance"]
    labels = [row[0] for row in cb["data"]]
    vals = [row[2] for row in cb["data"]]
    if labels:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.bar(labels, vals)
        ax.set_title("Class Balance")
        ax.set_ylabel("Objects")
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()
        fig.savefig(os.path.join(stats_dir, "class_balance.png"), dpi=160)
        plt.close(fig)

    # class_cooccurrence.png
    cc = artifacts["class_cooccurrence"]
    labels = cc["labels"]
    matrix = cc["data"]
    if labels and matrix:
        fig, ax = plt.subplots(figsize=(6.8, 6.0))
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_title("Class Co-occurrence")
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(stats_dir, "class_cooccurrence.png"), dpi=160)
        plt.close(fig)

    # class_sizes.png (average area per class)
    cs = artifacts["class_sizes"]
    labels = [row[0] for row in cs["data"]]
    avg_area = [row[2] for row in cs["data"]]
    if labels:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.bar(labels, avg_area)
        ax.set_title("Class Sizes (Avg area %)")
        ax.set_ylabel("Area (%)")
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()
        fig.savefig(os.path.join(stats_dir, "class_sizes.png"), dpi=160)
        plt.close(fig)

    # objects_distribution.png (class-wise heatmap)
    od = artifacts["objects_distribution"]
    x_labels = od.get("x_labels", [])
    y_labels = od.get("y_labels", [])
    z = od.get("z", [])
    if x_labels and y_labels and z:
        fig, ax = plt.subplots(figsize=(12, 6.4))
        im = ax.imshow(z, aspect="auto", cmap="viridis")
        ax.set_title("Objects on images - distribution for every class")
        ax.set_xlabel("Objects per image")
        ax.set_ylabel("Class")
        # Keep x ticks sparse for readability on long tails.
        step = max(1, len(x_labels) // 20)
        tick_idx = list(range(0, len(x_labels), step))
        ax.set_xticks(tick_idx, [x_labels[i] for i in tick_idx])
        ax.set_yticks(range(len(y_labels)), y_labels)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(stats_dir, "objects_distribution.png"), dpi=160)
        plt.close(fig)

    # classes_per_image.png
    cpi = artifacts["classes_per_image"]
    xs = [int(p["x"]) for p in cpi["series"][0]["data"]]
    ys = [p["y"] for p in cpi["series"][0]["data"]]
    if xs:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.bar(xs, ys)
        ax.set_title("Distinct classes per image")
        ax.set_xlabel("Classes")
        ax.set_ylabel("Images")
        fig.tight_layout()
        fig.savefig(os.path.join(stats_dir, "classes_per_image.png"), dpi=160)
        plt.close(fig)


def _build_preview_assets(config, output_dir, stats_data, samples_rel_dir="visualizations/samples", fast_selection=False):
    samples_rel = str(samples_rel_dir or "visualizations/samples").replace("\\", "/").strip("/")
    samples_dir = os.path.join(output_dir, *samples_rel.split("/"))
    vis_dir = os.path.dirname(samples_dir)
    annotations_dir = os.path.join(samples_dir, "annotations")
    _safe_makedirs(vis_dir)
    _safe_makedirs(samples_dir)
    _safe_makedirs(annotations_dir)

    # Remove legacy preview images only for the main global visualizations directory.
    if samples_rel == "visualizations/samples":
        for legacy_name in [
            "horizontal_grid.png",
            "vertical_grid.png",
            "side_annotations_grid.png",
            "poster.png",
        ]:
            legacy_path = os.path.join(vis_dir, legacy_name)
            if os.path.isfile(legacy_path):
                try:
                    os.remove(legacy_path)
                except Exception:
                    pass

    # Clear previous samples so reruns produce deterministic, fresh previews.
    for name in os.listdir(samples_dir):
        p = os.path.join(samples_dir, name)
        if os.path.isfile(p):
            try:
                os.remove(p)
            except Exception:
                pass
        elif os.path.isdir(p):
            try:
                shutil.rmtree(p)
            except Exception:
                pass

    _safe_makedirs(annotations_dir)

    selected = []
    sample_count = 0
    sample_cfg = config.get("sample_selection", {}) if isinstance(config, dict) else {}
    target_count = int(sample_cfg.get("count", 10))
    sample_seed = int(sample_cfg.get("seed", 42))
    resize_max_side = int(sample_cfg.get("resize_max_side", 1280))
    task_type = str(stats_data.get("task_type", "")).strip().lower()
    require_annotations = bool(
        sample_cfg.get("require_annotations", task_type in {"object_detection", "segmentation"})
    )

    if copy_selected_samples and fast_selection:
        selected = _quick_select_samples(
            stats_data,
            count=target_count,
            seed=sample_seed,
            require_annotations=require_annotations,
        )
        sample_count = copy_selected_samples(selected, samples_dir)
    elif select_diverse_samples and copy_selected_samples:
        selected = select_diverse_samples(
            stats_data,
            count=target_count,
            seed=sample_seed,
            require_annotations=require_annotations,
        )
        sample_count = copy_selected_samples(selected, samples_dir)

    sample_paths = []
    for name in sorted(os.listdir(samples_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            sample_paths.append(os.path.join(samples_dir, name))

    if sample_count == 0:
        sample_count = len(sample_paths)

    # Build slider-ready annotation pairs for object detection / segmentation samples.
    slider_pairs = 0
    splits = _resolved_splits(config)
    coco_cache = {}

    # Resize all sample images for visualization payloads.
    manifest_path = os.path.join(samples_dir, "samples_manifest.json")
    manifest_entries = []
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_payload = json.load(f)
            manifest_entries = manifest_payload.get("samples", [])
        except Exception:
            manifest_entries = []

    if task_type not in {"object_detection", "segmentation"}:
        for entry in manifest_entries:
            sample_file = entry.get("file", "")
            sample_path = os.path.join(samples_dir, sample_file)
            if not sample_file or not os.path.isfile(sample_path):
                continue
            resize_info = _resize_sample_image(sample_path, max_side=resize_max_side)
            if resize_info:
                entry["image_size"] = {
                    "width": int(resize_info["new_width"]),
                    "height": int(resize_info["new_height"]),
                }
                entry["resized_from"] = {
                    "width": int(resize_info["old_width"]),
                    "height": int(resize_info["old_height"]),
                }

    if task_type in {"object_detection", "segmentation"}:
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                payload = {"samples": [], "count": 0}

            entries = payload.get("samples", [])
            filtered_entries = []

            def _remove_sample_file_if_exists(sample_file):
                if not sample_file:
                    return
                sample_path = os.path.join(samples_dir, sample_file)
                if os.path.isfile(sample_path):
                    try:
                        os.remove(sample_path)
                    except Exception:
                        pass

            for idx, entry in enumerate(entries, start=1):
                src = entry.get("source", "")
                if not src or not os.path.isfile(src):
                    entry["slider_ready"] = False
                    entry["annotated_file"] = None
                    entry["annotation_file"] = None
                    if require_annotations:
                        _remove_sample_file_if_exists(entry.get("file", ""))
                        continue
                    filtered_entries.append(entry)
                    continue

                ann_info = _find_sample_annotation_info(src, config, splits, coco_cache)
                if not ann_info:
                    entry["slider_ready"] = False
                    entry["annotated_file"] = None
                    entry["annotation_file"] = None
                    if require_annotations:
                        _remove_sample_file_if_exists(entry.get("file", ""))
                        continue
                    filtered_entries.append(entry)
                    continue

                ann_canvas_size = _get_annotation_canvas_size(ann_info)

                raw_file = entry.get("file", "")
                raw_base = os.path.splitext(raw_file)[0] if raw_file else f"sample_{idx:02d}"

                sample_path = os.path.join(samples_dir, raw_file) if raw_file else ""
                resize_info = None
                if sample_path and os.path.isfile(sample_path):
                    resize_info = _resize_sample_image(
                        sample_path,
                        max_side=resize_max_side,
                        target_canvas_size=ann_canvas_size,
                    )
                    if resize_info:
                        entry["image_size"] = {
                            "width": int(resize_info["new_width"]),
                            "height": int(resize_info["new_height"]),
                        }
                        entry["resized_from"] = {
                            "width": int(resize_info["old_width"]),
                            "height": int(resize_info["old_height"]),
                        }
                        if resize_info.get("used_exif_orientation"):
                            entry["orientation_normalized"] = True

                new_w = int(resize_info.get("new_width", 0)) if resize_info else 0
                new_h = int(resize_info.get("new_height", 0)) if resize_info else 0

                if ann_canvas_size and ann_canvas_size[0] > 0 and ann_canvas_size[1] > 0 and new_w > 0 and new_h > 0:
                    sx = float(new_w) / float(ann_canvas_size[0])
                    sy = float(new_h) / float(ann_canvas_size[1])
                else:
                    sx = float(resize_info.get("scale_x", 1.0)) if resize_info else 1.0
                    sy = float(resize_info.get("scale_y", 1.0)) if resize_info else 1.0

                ann_copy_name = None
                ann_format = ann_info.get("format")

                if ann_format == "voc":
                    src_ann = ann_info.get("source_path", "")
                    ann_copy_name = f"{raw_base}_annotation.xml"
                    ann_copy_path = os.path.join(annotations_dir, ann_copy_name)
                    if not _scale_voc_annotation(src_ann, ann_copy_path, sx, sy, new_w, new_h):
                        ann_copy_name = None

                elif ann_format == "yolo":
                    src_ann = ann_info.get("source_path", "")
                    ann_copy_name = f"{raw_base}_annotation.txt"
                    ann_copy_path = os.path.join(annotations_dir, ann_copy_name)
                    if not _copy_or_scale_yolo_annotation(src_ann, ann_copy_path):
                        ann_copy_name = None

                elif ann_format == "coco" and ann_info.get("coco_record") is not None:
                    ann_copy_name = f"{raw_base}_annotation.json"
                    ann_copy_path = os.path.join(annotations_dir, ann_copy_name)
                    try:
                        scaled_record = _scale_coco_record(
                            ann_info.get("coco_record", {}),
                            sx,
                            sy,
                            new_w,
                            new_h,
                        )
                        with open(ann_copy_path, "w", encoding="utf-8") as f:
                            json.dump(scaled_record, f, indent=2)
                    except Exception:
                        ann_copy_name = None

                elif ann_format == "png_masks":
                    src_ann = ann_info.get("source_path", "")
                    ann_copy_name = f"{raw_base}_annotation.polygons.json"
                    ann_copy_path = os.path.join(annotations_dir, ann_copy_name)
                    if not _mask_to_polygons(
                        src_ann,
                        ann_copy_path,
                        target_size=(new_w, new_h) if new_w and new_h else None,
                        class_map=ann_info.get("class_map"),
                        class_exclude=ann_info.get("class_exclude"),
                        min_component_area_px=ann_info.get("min_component_area_px", 1),
                    ):
                        ann_copy_name = None

                entry["annotation_mode"] = ann_info.get("mode")
                if ann_format == "png_masks":
                    entry["annotation_format"] = "polygon"
                else:
                    entry["annotation_format"] = ann_format
                entry["annotated_file"] = None
                entry["annotation_file"] = (
                    f"annotations/{ann_copy_name}".replace("\\", "/") if ann_copy_name else None
                )
                entry["class_map"] = dict(ann_info.get("class_map", {}) or {})
                entry["slider_ready"] = bool(ann_copy_name)
                if ann_copy_name:
                    slider_pairs += 1

                if require_annotations and not entry["slider_ready"]:
                    _remove_sample_file_if_exists(entry.get("file", ""))
                    continue

                filtered_entries.append(entry)

            payload["samples"] = filtered_entries
            payload["count"] = len(filtered_entries)
            payload["slider_pairs"] = slider_pairs
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

    elif os.path.isfile(manifest_path):
        # Write back resized dimensions for non-OD/SEG tasks.
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            payload = {"samples": [], "count": 0}
        payload["samples"] = manifest_entries
        payload["count"] = len(manifest_entries)
        payload["slider_pairs"] = 0
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return sample_count, slider_pairs


def _attach_country_view_assets(config, output_dir, stats_data, country_views, all_sample_count, all_slider_pairs, all_spatial_meta):
    views = country_views.get("views", {}) if isinstance(country_views, dict) else {}
    view_order = country_views.get("view_order", []) if isinstance(country_views, dict) else []

    if "all" in views:
        views["all"]["samples_summary"] = {
            "sample_count": all_sample_count,
            "slider_pairs": all_slider_pairs,
            "samples_dir": "visualizations/samples",
            "samples_manifest": "visualizations/samples/samples_manifest.json",
            "sample_annotations_dir": "visualizations/samples/annotations",
        }
        views["all"]["spatial_heatmaps"] = all_spatial_meta

    for view_id in view_order:
        if view_id == "all":
            continue
        view = views.get(view_id)
        if not isinstance(view, dict):
            continue

        split_names = view.get("split_names", [])
        scoped_stats = _build_scoped_stats_data(stats_data, split_names)

        samples_rel = f"visualizations/country_views/{view_id}/samples"
        spatial_rel = f"visualizations/country_views/{view_id}/spatial_heatmaps"

        sample_count, slider_pairs = _build_preview_assets(
            config,
            output_dir,
            scoped_stats,
            samples_rel_dir=samples_rel,
            fast_selection=True,
        )
        spatial_meta = _build_spatial_heatmaps(
            scoped_stats,
            output_dir,
            relative_dir=spatial_rel,
        )

        view["samples_summary"] = {
            "sample_count": sample_count,
            "slider_pairs": slider_pairs,
            "samples_dir": samples_rel,
            "samples_manifest": f"{samples_rel}/samples_manifest.json",
            "sample_annotations_dir": f"{samples_rel}/annotations",
        }
        view["spatial_heatmaps"] = spatial_meta


def _attach_region_view_assets(config, output_dir, stats_data, region_views, all_sample_count, all_slider_pairs, all_spatial_meta):
    views = region_views.get("views", {}) if isinstance(region_views, dict) else {}
    view_order = region_views.get("view_order", []) if isinstance(region_views, dict) else []

    if "all" in views:
        views["all"]["samples_summary"] = {
            "sample_count": all_sample_count,
            "slider_pairs": all_slider_pairs,
            "samples_dir": "visualizations/samples",
            "samples_manifest": "visualizations/samples/samples_manifest.json",
            "sample_annotations_dir": "visualizations/samples/annotations",
        }
        views["all"]["spatial_heatmaps"] = all_spatial_meta

    for view_id in view_order:
        if view_id == "all":
            continue
        view = views.get(view_id)
        if not isinstance(view, dict):
            continue

        split_names = view.get("split_names", [])
        scoped_stats = _build_scoped_stats_data(stats_data, split_names)

        samples_rel = f"visualizations/region_views/{view_id}/samples"
        spatial_rel = f"visualizations/region_views/{view_id}/spatial_heatmaps"

        sample_count, slider_pairs = _build_preview_assets(
            config,
            output_dir,
            scoped_stats,
            samples_rel_dir=samples_rel,
            fast_selection=True,
        )
        spatial_meta = _build_spatial_heatmaps(
            scoped_stats,
            output_dir,
            relative_dir=spatial_rel,
        )

        view["samples_summary"] = {
            "sample_count": sample_count,
            "slider_pairs": slider_pairs,
            "samples_dir": samples_rel,
            "samples_manifest": f"{samples_rel}/samples_manifest.json",
            "sample_annotations_dir": f"{samples_rel}/annotations",
        }
        view["spatial_heatmaps"] = spatial_meta


def build_artifacts(config, output_dir, stats_data):
    stats_dir = os.path.join(output_dir, "stats")
    _safe_makedirs(stats_dir)

    artifacts = {
        "class_balance": _build_class_balance(stats_data),
        "class_cooccurrence": _build_class_cooccurrence(stats_data),
        "class_sizes": _build_class_sizes(stats_data),
        "objects_distribution": _build_objects_distribution(stats_data),
        "classes_per_image": _build_classes_per_image(stats_data),
        "classes_treemap": _build_classes_treemap(stats_data),
    }

    for name, payload in artifacts.items():
        _write_json(os.path.join(stats_dir, f"{name}.json"), payload)

    stats_manifest = {
        "version": 1,
        "artifacts": [
            {"name": "class_balance", "file": "stats/class_balance.json", "type": "table"},
            {"name": "class_cooccurrence", "file": "stats/class_cooccurrence.json", "type": "matrix"},
            {"name": "objects_distribution", "file": "stats/objects_distribution.json", "type": "heatmap"},
            {"name": "class_sizes", "file": "stats/class_sizes.json", "type": "table"},
            {"name": "classes_treemap", "file": "stats/classes_treemap.json", "type": "treemap"},
            {"name": "classes_per_image", "file": "stats/classes_per_image.json", "type": "histogram"},
        ],
    }

    spatial_meta = _build_spatial_heatmaps(stats_data, output_dir)
    _write_json(os.path.join(stats_dir, "spatial_heatmaps.json"), spatial_meta)
    stats_manifest["artifacts"].append(
        {"name": "spatial_heatmaps", "file": "stats/spatial_heatmaps.json", "type": "images"}
    )

    _plot_pngs(stats_dir, artifacts)
    sample_count, slider_pairs = _build_preview_assets(config, output_dir, stats_data)

    country_views = _build_country_views(config, stats_data)
    country_views_path = os.path.join(stats_dir, "country_views.json")
    country_views_dir = os.path.join(output_dir, "visualizations", "country_views")
    if country_views:
        _attach_country_view_assets(
            config,
            output_dir,
            stats_data,
            country_views,
            all_sample_count=sample_count,
            all_slider_pairs=slider_pairs,
            all_spatial_meta=spatial_meta,
        )
        _write_json(country_views_path, country_views)
        stats_manifest["artifacts"].append(
            {"name": "country_views", "file": "stats/country_views.json", "type": "drilldown"}
        )
    else:
        if os.path.isfile(country_views_path):
            try:
                os.remove(country_views_path)
            except Exception:
                pass
        if os.path.isdir(country_views_dir):
            try:
                shutil.rmtree(country_views_dir)
            except Exception:
                pass

    region_views = _build_region_views(config, stats_data)
    region_views_path = os.path.join(stats_dir, "region_views.json")
    region_views_dir = os.path.join(output_dir, "visualizations", "region_views")
    if region_views:
        _attach_region_view_assets(
            config,
            output_dir,
            stats_data,
            region_views,
            all_sample_count=sample_count,
            all_slider_pairs=slider_pairs,
            all_spatial_meta=spatial_meta,
        )
        _write_json(region_views_path, region_views)
        stats_manifest["artifacts"].append(
            {"name": "region_views", "file": "stats/region_views.json", "type": "drilldown"}
        )
    else:
        if os.path.isfile(region_views_path):
            try:
                os.remove(region_views_path)
            except Exception:
                pass
        if os.path.isdir(region_views_dir):
            try:
                shutil.rmtree(region_views_dir)
            except Exception:
                pass

    # Make sample count discoverable for web clients.
    _write_json(
        os.path.join(stats_dir, "samples_summary.json"),
        {
            "sample_count": sample_count,
            "slider_pairs": slider_pairs,
            "samples_dir": "visualizations/samples",
            "samples_manifest": "visualizations/samples/samples_manifest.json",
            "sample_annotations_dir": "visualizations/samples/annotations",
        },
    )
    stats_manifest["artifacts"].append(
        {"name": "samples_summary", "file": "stats/samples_summary.json", "type": "summary"}
    )
    _write_json(os.path.join(stats_dir, "manifest.json"), stats_manifest)

    print("✓ Built datasetninja-style stats and preview artifacts")
