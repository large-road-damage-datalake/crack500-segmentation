#!/usr/bin/env python3
"""
Select diverse sample images for dataset previews.

Selection strategy balances:
- class coverage (at least one image per class when possible),
- composition variety (single-class and multi-class images),
- visual diversity (greedy max-min over color-histogram features),
- reproducibility (seeded randomness).
"""

import argparse
import json
import math
import os
import random
import shutil
from collections import defaultdict


try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


NON_CRACK_HINTS = {
    "noncrack",
    "no_crack",
    "no-crack",
    "no crack",
    "non_crack",
    "non-crack",
    "non crack",
    "normal",
    "intact",
    "background",
    "negative",
    "uncracked",
    "undamaged",
    "sound",
    "healthy",
    "good",
    "none",
    "n",
}


def _vector_distance(a, b):
    if len(a) != len(b):
        return 0.0
    s = 0.0
    for i in range(len(a)):
        d = a[i] - b[i]
        s += d * d
    return math.sqrt(s)


def _image_feature(path, bins=8):
    """Compute a compact RGB histogram feature vector in [0, 1]."""
    if Image is None:
        return None
    try:
        with Image.open(path) as img:
            img = img.convert("RGB").resize((48, 48))
            px = list(img.getdata())
    except Exception:
        return None

    if not px:
        return None

    hist_r = [0] * bins
    hist_g = [0] * bins
    hist_b = [0] * bins
    for r, g, b in px:
        hist_r[min(bins - 1, int((r / 256.0) * bins))] += 1
        hist_g[min(bins - 1, int((g / 256.0) * bins))] += 1
        hist_b[min(bins - 1, int((b / 256.0) * bins))] += 1

    total = float(len(px))
    vec = []
    for h in (hist_r, hist_g, hist_b):
        vec.extend([v / total for v in h])
    return vec


def _candidate_records(stats_data):
    raw = stats_data.get("global", {}).get("_raw", {})
    image_paths = raw.get("image_paths", [])
    class_sets = raw.get("class_sets_per_image", [])
    class_counts = raw.get("class_counts_per_image", [])

    n = min(len(image_paths), len(class_sets), len(class_counts))
    records = []
    for i in range(n):
        p = image_paths[i]
        if not p or not os.path.exists(p):
            continue
        cset = sorted(set(class_sets[i] or []))
        ccounts = class_counts[i] or {}
        records.append(
            {
                "path": p,
                "classes": cset,
                "class_count": len(cset),
                "objects": int(sum(ccounts.values())) if isinstance(ccounts, dict) else 0,
            }
        )
    return records


def _normalize_label(label):
    s = str(label or "").strip().lower()
    s = s.replace("-", " ").replace("_", " ")
    s = " ".join(s.split())
    return s


def _is_non_crack_label(label):
    s = _normalize_label(label)
    compact = s.replace(" ", "")
    if s in NON_CRACK_HINTS or compact in {h.replace(" ", "").replace("-", "").replace("_", "") for h in NON_CRACK_HINTS}:
        return True
    if "non" in s and "crack" in s:
        return True
    if "no" in s and "crack" in s:
        return True
    return False


def _select_general(candidates, count, rng):
    """General-purpose diverse selection with class coverage + visual diversity."""
    if not candidates:
        return []
    count = max(1, min(count, len(candidates)))

    by_class = defaultdict(list)
    single_class = []
    multi_class = []
    for rec in candidates:
        if rec["class_count"] == 1:
            single_class.append(rec)
        elif rec["class_count"] > 1:
            multi_class.append(rec)
        for cls in rec["classes"]:
            by_class[cls].append(rec)

    selected = []
    seen_paths = set()

    def add_record(rec):
        p = rec["path"]
        if p in seen_paths:
            return False
        selected.append(rec)
        seen_paths.add(p)
        return True

    # 1) Ensure class coverage first.
    class_order = sorted(by_class.keys())
    rng.shuffle(class_order)
    for cls in class_order:
        cls_recs = by_class.get(cls, [])
        if not cls_recs:
            continue
        add_record(rng.choice(cls_recs))
        if len(selected) >= count:
            return selected[:count]

    # 2) Ensure single- and multi-class examples where available.
    if single_class:
        add_record(rng.choice(single_class))
    if len(selected) < count and multi_class:
        add_record(rng.choice(multi_class))

    if len(selected) >= count:
        return selected[:count]

    # 3) Fill remainder via visual-diversity greedy selection.
    remaining = [r for r in candidates if r["path"] not in seen_paths]
    if not remaining:
        return selected[:count]

    features = {}
    for rec in remaining + selected:
        features[rec["path"]] = _image_feature(rec["path"])

    if all(features[r["path"]] is None for r in remaining):
        remaining.sort(key=lambda r: (r["class_count"], r["objects"], rng.random()), reverse=True)
        for rec in remaining:
            add_record(rec)
            if len(selected) >= count:
                break
        return selected[:count]

    while len(selected) < count and remaining:
        best_idx = 0
        best_score = -1.0
        for i, rec in enumerate(remaining):
            vec = features.get(rec["path"])
            if vec is None:
                score = -1.0
            elif not selected:
                score = 0.0
            else:
                min_d = None
                for srec in selected:
                    svec = features.get(srec["path"])
                    if svec is None:
                        continue
                    d = _vector_distance(vec, svec)
                    if min_d is None or d < min_d:
                        min_d = d
                score = min_d if min_d is not None else 0.0

            score += rec["class_count"] * 0.001
            if score > best_score:
                best_score = score
                best_idx = i

        add_record(remaining.pop(best_idx))

    return selected[:count]


def _select_classification_balanced(candidates, count, rng):
    """
    Classification policy requested by user:
    - total samples: 10 (default),
    - cracked: 6-7,
    - non-cracked: 3-4.
    """
    non_cracked = []
    cracked = []
    for rec in candidates:
        labels = rec.get("classes", [])
        if labels and all(_is_non_crack_label(c) for c in labels):
            non_cracked.append(rec)
        else:
            cracked.append(rec)

    if not cracked or not non_cracked:
        return _select_general(candidates, count, rng)

    # Preferred split for total=10: 6 cracked + 4 non-cracked.
    non_target_low = max(1, int(round(count * 0.30)))
    non_target_high = max(non_target_low, int(round(count * 0.40)))

    if len(non_cracked) >= non_target_high:
        non_target = non_target_high
    elif len(non_cracked) >= non_target_low:
        non_target = non_target_low
    else:
        non_target = len(non_cracked)

    cracked_target = max(0, count - non_target)

    selected_non = _select_general(non_cracked, min(non_target, len(non_cracked)), rng)
    selected_cracked = _select_general(cracked, min(cracked_target, len(cracked)), rng)

    selected = selected_cracked + selected_non
    seen = {r["path"] for r in selected}

    if len(selected) < count:
        leftovers = [r for r in candidates if r["path"] not in seen]
        fill = _select_general(leftovers, min(count - len(selected), len(leftovers)), rng)
        selected.extend(fill)

    rng.shuffle(selected)
    return selected[:count]


def select_diverse_samples(stats_data, count=10, seed=42, require_annotations=False):
    """
    Return selected candidate records from stats_data raw block.
    """
    rng = random.Random(seed)
    candidates = _candidate_records(stats_data)
    if not candidates:
        return []

    if require_annotations:
        candidates = [rec for rec in candidates if int(rec.get("objects", 0)) > 0]
        if not candidates:
            return []

    count = max(1, min(count, len(candidates)))
    task_type = str(stats_data.get("task_type", "")).strip().lower()

    if task_type == "classification":
        return _select_classification_balanced(candidates, count, rng)

    return _select_general(candidates, count, rng)


def copy_selected_samples(selected, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    manifest = []

    for i, rec in enumerate(selected, start=1):
        src = rec["path"]
        ext = os.path.splitext(src)[1].lower()
        classes_tag = "none"
        if rec["classes"]:
            classes_tag = "-".join([c.replace(" ", "_") for c in rec["classes"][:3]])
        dst_name = f"sample_{i:02d}_c{rec['class_count']}_{classes_tag}{ext}"
        dst = os.path.join(output_dir, dst_name)

        try:
            shutil.copy2(src, dst)
        except Exception:
            continue

        manifest.append(
            {
                "file": dst_name,
                "source": src,
                "classes": rec["classes"],
                "num_classes": rec["class_count"],
                "objects": rec["objects"],
            }
        )

    with open(os.path.join(output_dir, "samples_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"samples": manifest, "count": len(manifest)}, f, indent=2)

    return len(manifest)


def main():
    parser = argparse.ArgumentParser(description="Select diverse sample images from computed stats")
    parser.add_argument("--stats", required=True, help="Path to stats.json")
    parser.add_argument("--output", required=True, help="Output sample directory")
    parser.add_argument("--count", type=int, default=10, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    with open(args.stats, "r", encoding="utf-8") as f:
        stats_data = json.load(f)

    selected = select_diverse_samples(stats_data, count=args.count, seed=args.seed)
    n = copy_selected_samples(selected, args.output)
    print(f"Selected {n} diverse samples into {args.output}")


if __name__ == "__main__":
    main()
