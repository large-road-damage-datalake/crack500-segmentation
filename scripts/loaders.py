import glob
import json
import os
import struct
import xml.etree.ElementTree as ET
from collections import defaultdict


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _new_stats_base(num_images=0):
    return {
        "num_images": num_images,
        "num_annotations": 0,
        "class_distribution": defaultdict(int),
        "objects_per_image": [],
        "image_resolution": {"widths": [], "heights": []},
        "bbox_area_rel": [],
        "bbox_area_rel_by_class": defaultdict(list),
        "class_sets_per_image": [],
        "class_counts_per_image": [],
        "class_area_sum_per_image": [],
        "image_paths": [],
        "bbox_shapes_by_class": defaultdict(list),
    }


def get_image_size(file_path):
    """Return (width, height) for PNG/JPEG using standard library only."""
    try:
        with open(file_path, "rb") as f:
            head = f.read(24)
            if len(head) != 24:
                return None

            if head.startswith(b"\x89PNG\r\n\x1a\n"):
                f.seek(16)
                return struct.unpack(">II", f.read(8))

            if head.startswith(b"\xff\xd8"):
                f.seek(0)
                size = 2
                marker = 0
                while not (0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC)):
                    f.seek(size, 1)
                    byte = f.read(1)
                    while byte and ord(byte) == 0xFF:
                        byte = f.read(1)
                    if not byte:
                        return None
                    marker = ord(byte)
                    size = struct.unpack(">H", f.read(2))[0] - 2
                f.read(1)
                h, w = struct.unpack(">HH", f.read(4))
                return (w, h)
    except Exception:
        return None

    return None


def _collect_images(images_root):
    images = []
    for ext in IMAGE_EXTS:
        images.extend(glob.glob(os.path.join(images_root, f"*{ext}")))
    return sorted(images)


def _map_detection_class(raw_value, class_map, class_exclude):
    raw_str = str(raw_value)

    if raw_value in class_exclude or raw_str in class_exclude:
        return None

    candidates = [raw_value, raw_str]
    if isinstance(raw_value, str) and raw_value.isdigit():
        try:
            candidates.append(int(raw_value))
        except Exception:
            pass

    mapped = None
    for key in candidates:
        if key in class_map:
            mapped = class_map[key]
            break
        key_str = str(key)
        if key_str in class_map:
            mapped = class_map[key_str]
            break

    if mapped is None:
        mapped = raw_str

    mapped_str = str(mapped)
    if mapped in class_exclude or mapped_str in class_exclude:
        return None
    return mapped_str


def _map_mask_class(raw_value, class_map, class_exclude):
    raw_str = str(raw_value)

    # Allow excluding raw mask codes directly (e.g., "0", "23") before remapping.
    if raw_value in class_exclude or raw_str in class_exclude:
        return None

    candidates = [raw_value, raw_str]
    if isinstance(raw_value, str) and raw_value.isdigit():
        try:
            candidates.append(int(raw_value))
        except Exception:
            pass

    if ":" in raw_str:
        parts = raw_str.split(":")
        if all(p.isdigit() for p in parts):
            rgb = tuple(int(p) for p in parts)
            candidates.append(rgb)
            if len(set(rgb)) == 1:
                candidates.append(rgb[0])
                candidates.append(str(rgb[0]))

    mapped = None
    for key in candidates:
        if key in class_map:
            mapped = class_map[key]
            break
        key_str = str(key)
        if key_str in class_map:
            mapped = class_map[key_str]
            break

    if mapped is None:
        # Common binary-mask case: config maps one positive value (e.g. 255->crack),
        # but files may store a different non-zero foreground value (e.g. 1).
        if class_map and raw_str not in {"0", "0:0:0"} and len(class_map) == 1:
            try:
                only_val = next(iter(class_map.values()))
                mapped = str(only_val)
            except Exception:
                mapped = None

    if mapped is None:
        mapped = raw_str

    mapped_str = str(mapped)
    if mapped in class_exclude or mapped_str in class_exclude:
        return None
    return mapped_str


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


def _single_mask_label(class_map):
    if not isinstance(class_map, dict) or not class_map:
        return None
    labels = {str(v).strip() for v in class_map.values() if str(v).strip()}
    if len(labels) == 1:
        return next(iter(labels))
    return None


def _infer_binary_foreground_values(unique_values, unique_counts, class_map):
    if len(unique_values) != 2 or len(unique_counts) != 2:
        return None

    c0 = int(unique_counts[0])
    c1 = int(unique_counts[1])
    total = c0 + c1
    if total <= 0:
        return None

    min_idx = 0 if c0 <= c1 else 1
    minority_val = int(unique_values[min_idx])
    majority_val = int(unique_values[1 - min_idx])
    minority_ratio = float(unique_counts[min_idx]) / float(total)

    mapped_values = set()
    if isinstance(class_map, dict):
        for raw_key in class_map.keys():
            parsed = _parse_int_mask_value(raw_key)
            if parsed is not None:
                mapped_values.add(parsed)

    if len(mapped_values) == 1:
        only_mapped = next(iter(mapped_values))
        if only_mapped == minority_val:
            return {minority_val}
        if only_mapped == majority_val and minority_ratio <= 0.45:
            return {minority_val}

    single_label = _single_mask_label(class_map)
    if minority_ratio <= 0.45 and (not class_map or single_label is not None):
        return {minority_val}

    return None


def _build_class_masks_from_grayscale(base, class_map, class_exclude):
    try:
        import numpy as np
    except Exception:
        return {}

    class_masks = {}
    unique_vals, unique_counts = np.unique(base, return_counts=True)
    if unique_vals.size == 0:
        return class_masks
    if unique_vals.size == 1:
        return class_masks

    foreground_override = _infer_binary_foreground_values(
        unique_vals.tolist(),
        unique_counts.tolist(),
        class_map,
    )
    single_label = _single_mask_label(class_map)

    for raw in unique_vals:
        raw_i = int(raw)

        if foreground_override is not None and raw_i not in foreground_override:
            continue
        if foreground_override is None and raw_i == 0:
            continue

        mapped = _map_mask_class(raw_i, class_map, class_exclude)
        if foreground_override is not None and single_label and mapped in {str(raw_i), "0", "0:0:0"}:
            mapped = single_label
        if not mapped:
            continue

        binary = (base == raw)
        if mapped in class_masks:
            class_masks[mapped] = np.logical_or(class_masks[mapped], binary)
        else:
            class_masks[mapped] = binary

    return class_masks


def _connected_components_with_stats(binary_mask):
    """
    Return connected component stats for a binary mask.
    Each component is a tuple: (min_x, min_y, max_x, max_y, area_px).
    """
    try:
        import numpy as np
    except Exception:
        return []

    if binary_mask is None:
        return []

    mask = np.asarray(binary_mask, dtype=bool)
    if mask.ndim != 2:
        return []

    h, w = mask.shape
    if h <= 0 or w <= 0:
        return []

    visited = np.zeros_like(mask, dtype=bool)
    ys, xs = np.where(mask)
    components = []

    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue

        stack = [(y0, x0)]
        visited[y0, x0] = True

        min_x = max_x = x0
        min_y = max_y = y0
        area_px = 0

        while stack:
            y, x = stack.pop()
            area_px += 1

            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y

            for ny in (y - 1, y, y + 1):
                if ny < 0 or ny >= h:
                    continue
                for nx in (x - 1, x, x + 1):
                    if nx < 0 or nx >= w or (nx == x and ny == y):
                        continue
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

        components.append((min_x, min_y, max_x, max_y, area_px))

    return components


def _single_component_with_stats(binary_mask):
    """
    Summarize a binary mask as one component spanning all positive pixels.
    """
    try:
        import numpy as np
    except Exception:
        return []

    if binary_mask is None:
        return []

    mask = np.asarray(binary_mask, dtype=bool)
    if mask.ndim != 2:
        return []

    ys, xs = np.where(mask)
    if ys.size == 0:
        return []

    min_x = int(xs.min())
    max_x = int(xs.max())
    min_y = int(ys.min())
    max_y = int(ys.max())
    area_px = int(ys.size)
    return [(min_x, min_y, max_x, max_y, area_px)]


def _filter_components_by_area(components, min_component_area_px=1):
    try:
        min_area = int(min_component_area_px)
    except Exception:
        min_area = 1
    if min_area <= 1:
        return list(components or [])
    return [comp for comp in (components or []) if len(comp) >= 5 and int(comp[4]) >= min_area]


def load_coco_stats(annotations_path, images_root):
    # Support image-only splits (e.g., test sets without labels).
    if not os.path.exists(annotations_path):
        if not os.path.isdir(images_root):
            print(f"[{images_root}] is not a directory.")
            return None

        image_files = _collect_images(images_root)
        stats = _new_stats_base(num_images=len(image_files))
        for img_path in image_files:
            res = get_image_size(img_path)
            if res:
                stats["image_resolution"]["widths"].append(res[0])
                stats["image_resolution"]["heights"].append(res[1])
            stats["objects_per_image"].append(0)
            stats["class_sets_per_image"].append([])
            stats["class_counts_per_image"].append({})
            stats["class_area_sum_per_image"].append({})
            stats["image_paths"].append(img_path)
        return stats

    try:
        with open(annotations_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading COCO JSON: {e}")
        return None

    images = data.get("images", [])
    stats = _new_stats_base(num_images=len(images))
    cat_id_to_name = {c["id"]: c["name"] for c in data.get("categories", [])}

    img_res = {}
    img_paths = {}
    img_ann_count = defaultdict(int)
    img_class_sets = defaultdict(set)
    img_class_counts = defaultdict(lambda: defaultdict(int))
    img_class_area_sums = defaultdict(lambda: defaultdict(float))

    for img in images:
        w = img.get("width")
        h = img.get("height")
        img_res[img["id"]] = (w, h)
        file_name = img.get("file_name", "")
        img_paths[img["id"]] = os.path.join(images_root, file_name) if file_name else ""
        if w and h:
            stats["image_resolution"]["widths"].append(w)
            stats["image_resolution"]["heights"].append(h)

    for ann in data.get("annotations", []):
        cls = cat_id_to_name.get(ann.get("category_id"), str(ann.get("category_id")))
        image_id = ann.get("image_id")
        stats["class_distribution"][cls] += 1
        stats["num_annotations"] += 1
        img_ann_count[image_id] += 1
        img_class_sets[image_id].add(cls)
        img_class_counts[image_id][cls] += 1

        bbox = ann.get("bbox", [])
        if len(bbox) == 4:
            bx, by, bw, bh = bbox[0], bbox[1], bbox[2], bbox[3]
            iw, ih = img_res.get(image_id, (0, 0))
            if iw and ih:
                rel_area = (bw * bh) / (iw * ih)
                stats["bbox_area_rel"].append(rel_area)
                stats["bbox_area_rel_by_class"][cls].append(rel_area)
                img_class_area_sums[image_id][cls] += rel_area
                stats["bbox_shapes_by_class"][cls].append(
                    {
                        "area_rel": rel_area,
                        "width_rel": bw / iw,
                        "height_rel": bh / ih,
                        "width_px": bw,
                        "height_px": bh,
                        "cx_rel": (bx + bw / 2.0) / iw,
                        "cy_rel": (by + bh / 2.0) / ih,
                    }
                )

    for img in images:
        image_id = img["id"]
        stats["objects_per_image"].append(img_ann_count.get(image_id, 0))
        stats["class_sets_per_image"].append(sorted(list(img_class_sets.get(image_id, set()))))
        stats["class_counts_per_image"].append(dict(img_class_counts.get(image_id, {})))
        stats["class_area_sum_per_image"].append(dict(img_class_area_sums.get(image_id, {})))
        stats["image_paths"].append(img_paths.get(image_id, ""))

    return stats


def load_yolo_stats(images_root, annotations_root=None, class_map=None, class_exclude=None):
    if not os.path.isdir(images_root):
        print(f"[{images_root}] is not a directory.")
        return None

    class_map = class_map or {}
    class_exclude = set(class_exclude or [])
    image_files = _collect_images(images_root)
    stats = _new_stats_base(num_images=len(image_files))

    for img_path in image_files:
        res = get_image_size(img_path)
        if res:
            iw, ih = res
            stats["image_resolution"]["widths"].append(iw)
            stats["image_resolution"]["heights"].append(ih)

        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = None
        if annotations_root and os.path.isdir(annotations_root):
            label_path = os.path.join(annotations_root, basename + ".txt")

        count = 0
        class_set = set()
        class_count_map = defaultdict(int)
        class_area_sum_map = defaultdict(float)
        if label_path and os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = _map_detection_class(parts[0], class_map, class_exclude)
                if not cls:
                    continue
                class_set.add(cls)
                stats["class_distribution"][cls] += 1
                stats["num_annotations"] += 1
                count += 1
                class_count_map[cls] += 1

                try:
                    cx_rel = float(parts[1])
                    cy_rel = float(parts[2])
                    w_rel = float(parts[3])
                    h_rel = float(parts[4])
                    rel_area = w_rel * h_rel
                    stats["bbox_area_rel"].append(rel_area)
                    stats["bbox_area_rel_by_class"][cls].append(rel_area)
                    class_area_sum_map[cls] += rel_area
                    stats["bbox_shapes_by_class"][cls].append(
                        {
                            "area_rel": rel_area,
                            "width_rel": w_rel,
                            "height_rel": h_rel,
                            "width_px": (w_rel * iw) if iw else 0,
                            "height_px": (h_rel * ih) if ih else 0,
                            "cx_rel": cx_rel,
                            "cy_rel": cy_rel,
                        }
                    )
                except ValueError:
                    pass

        stats["objects_per_image"].append(count)
        stats["class_sets_per_image"].append(sorted(list(class_set)))
        stats["class_counts_per_image"].append(dict(class_count_map))
        stats["class_area_sum_per_image"].append(dict(class_area_sum_map))
        stats["image_paths"].append(img_path)

    return stats


def load_voc_stats(images_root, annotations_root=None, class_map=None, class_exclude=None):
    if not os.path.isdir(images_root):
        return None

    class_map = class_map or {}
    class_exclude = set(class_exclude or [])
    image_files = _collect_images(images_root)
    stats = _new_stats_base(num_images=len(image_files))

    for img_path in image_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        xml_path = ""
        if annotations_root and os.path.isdir(annotations_root):
            xml_path = os.path.join(annotations_root, basename + ".xml")

        res = get_image_size(img_path)
        iw, ih = (0, 0)
        if res:
            iw, ih = res
            stats["image_resolution"]["widths"].append(iw)
            stats["image_resolution"]["heights"].append(ih)

        count = 0
        class_set = set()
        class_count_map = defaultdict(int)
        class_area_sum_map = defaultdict(float)
        if xml_path and os.path.exists(xml_path):
            try:
                root = ET.parse(xml_path).getroot()
                if not iw or not ih:
                    size_node = root.find("size")
                    if size_node is not None:
                        try:
                            iw = int(float(size_node.findtext("width", default="0")))
                            ih = int(float(size_node.findtext("height", default="0")))
                            if iw and ih:
                                stats["image_resolution"]["widths"].append(iw)
                                stats["image_resolution"]["heights"].append(ih)
                        except Exception:
                            iw, ih = (0, 0)
                for obj in root.findall("object"):
                    name_node = obj.find("name")
                    if name_node is None:
                        continue
                    cls = class_map.get(name_node.text, name_node.text)
                    if cls in class_exclude:
                        continue
                    class_set.add(cls)
                    stats["class_distribution"][cls] += 1
                    stats["num_annotations"] += 1
                    count += 1
                    class_count_map[cls] += 1

                    bndbox = obj.find("bndbox")
                    if bndbox is not None and iw and ih:
                        try:
                            xmin = float(bndbox.find("xmin").text)
                            ymin = float(bndbox.find("ymin").text)
                            xmax = float(bndbox.find("xmax").text)
                            ymax = float(bndbox.find("ymax").text)
                            bw = max(0.0, xmax - xmin)
                            bh = max(0.0, ymax - ymin)
                            rel_area = (bw * bh) / (iw * ih)
                            stats["bbox_area_rel"].append(rel_area)
                            stats["bbox_area_rel_by_class"][cls].append(rel_area)
                            class_area_sum_map[cls] += rel_area
                            stats["bbox_shapes_by_class"][cls].append(
                                {
                                    "area_rel": rel_area,
                                    "width_rel": bw / iw,
                                    "height_rel": bh / ih,
                                    "width_px": bw,
                                    "height_px": bh,
                                    "cx_rel": (xmin + bw / 2.0) / iw,
                                    "cy_rel": (ymin + bh / 2.0) / ih,
                                }
                            )
                        except Exception:
                            pass
            except Exception:
                pass

        stats["objects_per_image"].append(count)
        stats["class_sets_per_image"].append(sorted(list(class_set)))
        stats["class_counts_per_image"].append(dict(class_count_map))
        stats["class_area_sum_per_image"].append(dict(class_area_sum_map))
        stats["image_paths"].append(img_path)

    return stats


def load_image_folder_stats(images_root, class_map=None, class_exclude=None):
    if not os.path.isdir(images_root):
        return None

    class_map = class_map or {}
    class_exclude = set(class_exclude or [])
    stats = _new_stats_base(num_images=0)

    class_dirs = [
        d for d in sorted(os.listdir(images_root))
        if os.path.isdir(os.path.join(images_root, d))
    ]

    for cls in class_dirs:
        if cls in class_exclude:
            continue
        class_path = os.path.join(images_root, cls)
        target_cls = class_map.get(cls, cls)
        class_images = []
        for root, _, files in os.walk(class_path):
            for fname in files:
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                    class_images.append(os.path.join(root, fname))
        class_images.sort()

        stats["class_distribution"][target_cls] += len(class_images)
        stats["num_images"] += len(class_images)
        stats["num_annotations"] += len(class_images)

        for img_path in class_images:
            res = get_image_size(img_path)
            if res:
                stats["image_resolution"]["widths"].append(res[0])
                stats["image_resolution"]["heights"].append(res[1])
            stats["objects_per_image"].append(1)
            stats["class_sets_per_image"].append([target_cls])
            stats["class_counts_per_image"].append({target_cls: 1})
            stats["class_area_sum_per_image"].append({})
            stats["image_paths"].append(img_path)

    return stats


def load_png_mask_stats(
    images_root,
    masks_root=None,
    class_map=None,
    class_exclude=None,
    mask_suffixes=None,
    connected_components=True,
    min_component_area_px=1,
):
    if not os.path.isdir(images_root):
        return None

    if not masks_root or not os.path.isdir(masks_root):
        masks_root = ""

    class_map = class_map or {}
    class_exclude = set(class_exclude or [])
    suffixes = mask_suffixes if isinstance(mask_suffixes, (list, tuple)) else [""]
    suffixes = [str(s) for s in suffixes if s is not None]
    if "" not in suffixes:
        suffixes.append("")

    if isinstance(connected_components, str):
        connected_components = connected_components.strip().lower() not in {"0", "false", "no", "off"}
    else:
        connected_components = bool(connected_components)

    try:
        min_component_area_px = int(min_component_area_px)
    except Exception:
        min_component_area_px = 1
    if min_component_area_px < 1:
        min_component_area_px = 1

    image_files = _collect_images(images_root)
    stats = _new_stats_base(num_images=len(image_files))

    try:
        from PIL import Image
        has_pil = True
    except ImportError:
        has_pil = False
        print("Warning: PIL not found. PNG mask class stats will be limited.")

    for img_path in image_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = None
        for suffix in suffixes:
            for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                candidate = os.path.join(masks_root, f"{basename}{suffix}{ext}")
                if os.path.exists(candidate):
                    mask_path = candidate
                    break
            if mask_path:
                break

        res = get_image_size(img_path)
        if res:
            stats["image_resolution"]["widths"].append(res[0])
            stats["image_resolution"]["heights"].append(res[1])

        class_set = set()
        count = 0
        class_count_map = defaultdict(int)
        class_area_sum_map = defaultdict(float)
        if mask_path:
            if has_pil:
                try:
                    import numpy as np

                    with Image.open(mask_path) as mask_img:
                        mask_arr = np.array(mask_img)

                    mask_h, mask_w = (0, 0)
                    if mask_arr.ndim >= 2:
                        mask_h, mask_w = int(mask_arr.shape[0]), int(mask_arr.shape[1])

                    if mask_h > 0 and mask_w > 0:
                        image_area = float(mask_w * mask_h)
                        class_masks = {}

                        if mask_arr.ndim == 2:
                            class_masks = _build_class_masks_from_grayscale(
                                mask_arr,
                                class_map,
                                class_exclude,
                            )
                        elif mask_arr.ndim == 3:
                            channels = int(mask_arr.shape[2])
                            if channels > 0:
                                active_channels = min(channels, 4)

                                # Fast path: RGB/RGBA masks where channels carry identical grayscale values.
                                base = mask_arr[:, :, 0]
                                grayscale_like = True
                                for ci in range(1, min(active_channels, 3)):
                                    if not np.array_equal(base, mask_arr[:, :, ci]):
                                        grayscale_like = False
                                        break

                                if grayscale_like:
                                    class_masks = _build_class_masks_from_grayscale(
                                        base,
                                        class_map,
                                        class_exclude,
                                    )
                                else:
                                    # Generic color mask path without expensive np.unique(..., axis=0).
                                    channels_arr = mask_arr[:, :, :active_channels].astype(np.uint32, copy=False)
                                    packed = channels_arr[:, :, 0].copy()
                                    for ci in range(1, active_channels):
                                        packed = (packed << 8) | channels_arr[:, :, ci]

                                    for code in np.unique(packed):
                                        code_i = int(code)
                                        color_vals = [0] * active_channels
                                        tmp = code_i
                                        for i in range(active_channels - 1, -1, -1):
                                            color_vals[i] = tmp & 0xFF
                                            tmp >>= 8
                                        color_tuple = tuple(int(v) for v in color_vals)
                                        if all(v == 0 for v in color_tuple):
                                            continue

                                        key = ":".join(str(v) for v in color_tuple)
                                        mapped = _map_mask_class(key, class_map, class_exclude)
                                        if not mapped and active_channels >= 3:
                                            key_rgb = ":".join(str(v) for v in color_tuple[:3])
                                            mapped = _map_mask_class(key_rgb, class_map, class_exclude)
                                        if not mapped:
                                            continue

                                        binary = (packed == code_i)
                                        if mapped in class_masks:
                                            class_masks[mapped] = np.logical_or(class_masks[mapped], binary)
                                        else:
                                            class_masks[mapped] = binary

                        for cls, cls_mask in class_masks.items():
                            if connected_components:
                                components = _connected_components_with_stats(cls_mask)
                            else:
                                components = _single_component_with_stats(cls_mask)
                            components = _filter_components_by_area(
                                components,
                                min_component_area_px=min_component_area_px,
                            )
                            if not components:
                                continue

                            class_set.add(cls)
                            n_comp = len(components)
                            class_count_map[cls] += n_comp
                            stats["class_distribution"][cls] += n_comp
                            stats["num_annotations"] += n_comp
                            count += n_comp

                            for min_x, min_y, max_x, max_y, area_px in components:
                                bw = max(1.0, float(max_x - min_x + 1))
                                bh = max(1.0, float(max_y - min_y + 1))
                                rel_area = float(area_px) / image_area
                                class_area_sum_map[cls] += rel_area
                                stats["bbox_area_rel"].append(rel_area)
                                stats["bbox_area_rel_by_class"][cls].append(rel_area)
                                stats["bbox_shapes_by_class"][cls].append(
                                    {
                                        "area_rel": rel_area,
                                        "width_rel": bw / float(mask_w),
                                        "height_rel": bh / float(mask_h),
                                        "width_px": bw,
                                        "height_px": bh,
                                        "cx_rel": (float(min_x) + bw / 2.0) / float(mask_w),
                                        "cy_rel": (float(min_y) + bh / 2.0) / float(mask_h),
                                    }
                                )
                except Exception:
                    # Fallback to image-level class presence if numpy/pixel parsing fails.
                    try:
                        with Image.open(mask_path) as mask:
                            colors = mask.getcolors(maxcolors=1024)
                        if colors is None:
                            colors = []
                        for _, val in colors:
                            if isinstance(val, tuple):
                                val = ":".join(map(str, val))
                            key = str(val)
                            if key == "0" or key == "0:0:0":
                                continue
                            mapped = _map_mask_class(key, class_map, class_exclude)
                            if mapped:
                                class_set.add(mapped)
                        count = len(class_set)
                        for cls in class_set:
                            stats["class_distribution"][cls] += 1
                            class_count_map[cls] = 1
                        stats["num_annotations"] += count
                    except Exception:
                        pass
            else:
                class_set.add("unknown")
                count = 1
                stats["class_distribution"]["unknown"] += 1
                stats["num_annotations"] += 1
                class_count_map["unknown"] = 1

        stats["objects_per_image"].append(count)
        stats["class_sets_per_image"].append(sorted(list(class_set)))
        stats["class_counts_per_image"].append(dict(class_count_map))
        stats["class_area_sum_per_image"].append(dict(class_area_sum_map))
        stats["image_paths"].append(img_path)

    return stats
