#!/usr/bin/env python3
"""
Strict validator for dataset package completeness.
"""

import json
import os
import re
import shutil
from collections import defaultdict


REQUIRED_FILES = [
    "METADATA.json",
    "README.md",
    "ABSTRACT.md",
    "SUMMARY.md",
    "CITATION.bib",
    "CITATION.md",
    "DOWNLOAD.md",
    "LICENSE.md",
    "stats/stats.json",
    "stats/class_balance.json",
    "stats/class_cooccurrence.json",
    "stats/class_sizes.json",
    "stats/objects_distribution.json",
    "stats/classes_per_image.json",
    "stats/classes_treemap.json",
    "visualizations/samples/samples_manifest.json",
]

REQUIRED_METADATA_FIELDS = [
    "basic_info.id",
    "basic_info.name",
    "basic_info.short_name",
    "basic_info.description",
    "project_context.task",
    "statistics.n_images",
    "statistics.n_annotations",
    "license",
    "links.github",
    "links.download",
    "citation.bibtex",
]


def _get_nested(obj, dotted):
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _normalized_label(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _is_code_label(name):
    token = re.sub(r"\s+", "", str(name or ""))
    return bool(re.match(r"^[A-Za-z]+[0-9][A-Za-z0-9]*$", token))


_ROLE_TO_SEMANTIC_ROLE = {
    "damage_target": "damage",
    "positive": "damage",
    "repair": "repair",
    "context": "road_feature",
    "negative": "absence",
    "background": "none",
    "ignore": "none",
    "unknown": "unknown",
    "code_legacy": "unknown",
}

_ROLE_TO_TRAINING_ROLE = {
    "damage_target": "target",
    "positive": "target",
    "repair": "context",
    "context": "context",
    "negative": "negative",
    "background": "background",
    "ignore": "ignore",
    "unknown": "ignore",
    "code_legacy": "ignore",
}

_ALLOWED_MAPPING_TYPES = {"exact", "normalized", "inferred", "codebook", "codebook-based", "lossy"}
_ALLOWED_MAPPING_CONFIDENCE = {"high", "medium", "low"}
_ALLOWED_MAPPING_STATUS = {"draft", "verified", "deprecated"}


def _derive_semantic_role(role, canonical_name=""):
    role_norm = str(role or "").strip().lower()
    canonical = str(canonical_name or "").strip().lower()
    if role_norm in _ROLE_TO_SEMANTIC_ROLE:
        return _ROLE_TO_SEMANTIC_ROLE[role_norm]
    if canonical == "background":
        return "none"
    if canonical in {"ignore", "unlabeled"}:
        return "none"
    if canonical.startswith("unknown."):
        return "unknown"
    if canonical.startswith("class.negative"):
        return "absence"
    if canonical.startswith("context."):
        return "road_feature"
    if canonical.startswith("repair.") or canonical.startswith("patch."):
        return "repair"
    if canonical.startswith("crack") or canonical.startswith("pothole") or canonical.startswith("damage."):
        return "damage"
    return "unknown"


def _derive_training_role(role, canonical_name=""):
    role_norm = str(role or "").strip().lower()
    canonical = str(canonical_name or "").strip().lower()
    if role_norm in _ROLE_TO_TRAINING_ROLE:
        return _ROLE_TO_TRAINING_ROLE[role_norm]
    if canonical == "background":
        return "background"
    if canonical in {"ignore", "unlabeled"}:
        return "ignore"
    if canonical.startswith("unknown."):
        return "ignore"
    if canonical.startswith("class.negative"):
        return "negative"
    if canonical.startswith("context."):
        return "context"
    if canonical.startswith("repair.") or canonical.startswith("patch."):
        return "context"
    if canonical.startswith("crack") or canonical.startswith("pothole") or canonical.startswith("damage."):
        return "target"
    return "ignore"


def _validate_class_taxonomy_semantics(metadata):
    issues = []

    project_context = metadata.get("project_context", {}) if isinstance(metadata, dict) else {}
    declared_damage_types = project_context.get("damage_types", [])
    if not isinstance(declared_damage_types, list):
        declared_damage_types = []

    class_taxonomy = metadata.get("class_taxonomy", {}) if isinstance(metadata, dict) else {}
    taxonomy_by_canonical = {}
    damage_target_families = set()

    if isinstance(class_taxonomy, dict):
        version = class_taxonomy.get("version")
        is_v2 = str(version or "").strip().lower().startswith("road_damage_v2")
        classes = class_taxonomy.get("classes")
        if version in (None, ""):
            issues.append("class_taxonomy.version missing")
        if not isinstance(classes, list) or not classes:
            issues.append("class_taxonomy.classes missing or empty")
            classes = []

        for ci, cls in enumerate(classes):
            if not isinstance(cls, dict):
                continue

            canonical_name = str(cls.get("canonical_name", "")).strip()
            role = str(cls.get("role", "")).strip()
            semantic_role = str(cls.get("semantic_role", "")).strip()
            training_role = str(cls.get("training_role", "")).strip()
            damage_family = str(cls.get("damage_family", "")).strip()

            if not canonical_name:
                issues.append(f"class_taxonomy.classes[{ci}].canonical_name missing")
                continue
            taxonomy_by_canonical[canonical_name] = cls

            if not role:
                issues.append(f"class_taxonomy.classes[{ci}].role missing for canonical '{canonical_name}'")
            if is_v2 and not semantic_role:
                issues.append(
                    f"class_taxonomy.classes[{ci}].semantic_role missing for canonical '{canonical_name}'"
                )
            if is_v2 and not training_role:
                issues.append(
                    f"class_taxonomy.classes[{ci}].training_role missing for canonical '{canonical_name}'"
                )

            role_norm = role.lower()
            semantic_norm = (semantic_role or _derive_semantic_role(role, canonical_name)).lower()
            training_norm = (training_role or _derive_training_role(role, canonical_name)).lower()
            is_damage_target = (
                (semantic_norm == "damage" and training_norm == "target")
                or role_norm in {"damage_target", "positive"}
            )

            if is_damage_target:
                if not damage_family or damage_family.lower() == "unknown":
                    issues.append(
                        f"class_taxonomy.classes[{ci}].damage_family required for damage target canonical '{canonical_name}'"
                    )
                else:
                    damage_target_families.add(damage_family)

            canonical_norm = canonical_name.strip().lower()
            if canonical_norm.startswith("unknown.") and training_norm in {"background", "target"}:
                issues.append(
                    f"class_taxonomy.classes[{ci}] unknown canonical '{canonical_name}' cannot use training_role '{training_norm}'"
                )

            if canonical_norm == "background":
                source_labels = cls.get("source_labels") or cls.get("aliases") or []
                if isinstance(source_labels, list):
                    conflated = [
                        s
                        for s in source_labels
                        if any(tok in _normalized_label(s) for tok in ("unknown", "unlabeled", "ignore"))
                    ]
                    if conflated:
                        issues.append(
                            "background canonical is conflated with unknown/ignore/unlabeled labels: "
                            + ", ".join(sorted(set(str(v) for v in conflated)))
                        )

            if canonical_norm == "ignore" and training_norm != "ignore":
                issues.append(
                    f"class_taxonomy.classes[{ci}] canonical 'ignore' must use training_role='ignore'"
                )
            if canonical_norm == "unlabeled" and training_norm not in {"ignore", "negative"}:
                issues.append(
                    f"class_taxonomy.classes[{ci}] canonical 'unlabeled' must use training_role ignore/negative"
                )

            mapping_type = str(cls.get("mapping_type", "")).strip().lower()
            mapping_confidence = str(cls.get("mapping_confidence", "")).strip().lower()
            mapping_status = str(cls.get("mapping_status", "")).strip().lower()
            if mapping_type and mapping_type not in _ALLOWED_MAPPING_TYPES:
                issues.append(
                    f"class_taxonomy.classes[{ci}] invalid mapping_type '{mapping_type}' for canonical '{canonical_name}'"
                )
            if mapping_confidence and mapping_confidence not in _ALLOWED_MAPPING_CONFIDENCE:
                issues.append(
                    f"class_taxonomy.classes[{ci}] invalid mapping_confidence '{mapping_confidence}' for canonical '{canonical_name}'"
                )
            if mapping_status and mapping_status not in _ALLOWED_MAPPING_STATUS:
                issues.append(
                    f"class_taxonomy.classes[{ci}] invalid mapping_status '{mapping_status}' for canonical '{canonical_name}'"
                )
            if mapping_type and not mapping_confidence:
                issues.append(
                    f"class_taxonomy.classes[{ci}] mapping_type set but mapping_confidence missing for canonical '{canonical_name}'"
                )

            if training_norm == "target" and semantic_norm != "damage":
                issues.append(
                    f"class_taxonomy.classes[{ci}] target training_role requires semantic_role=damage for canonical '{canonical_name}'"
                )

            if role_norm in {"damage_target", "positive"} and damage_family:
                damage_target_families.add(damage_family)
    else:
        issues.append("class_taxonomy missing or invalid")

    schemas = metadata.get("annotation_schema", []) if isinstance(metadata, dict) else []
    if not isinstance(schemas, list):
        schemas = []

    for si, schema in enumerate(schemas):
        if not isinstance(schema, dict):
            continue
        classes = schema.get("classes", [])
        if not isinstance(classes, list):
            continue

        for ci, cls in enumerate(classes):
            if not isinstance(cls, dict):
                continue

            label = str(cls.get("name", "")).strip() or f"class_{si}_{ci}"
            canonical_name = str(cls.get("canonical_name", "")).strip()

            if not canonical_name:
                issues.append(
                    f"annotation_schema[{si}].classes[{ci}].canonical_name missing for label '{label}'"
                )
            elif canonical_name not in taxonomy_by_canonical:
                issues.append(
                    f"annotation_schema[{si}].classes[{ci}] canonical_name '{canonical_name}' not found in class_taxonomy.classes"
                )

            if _is_code_label(label):
                taxonomy_cls = taxonomy_by_canonical.get(canonical_name, {})
                role_norm = str(taxonomy_cls.get("role", "")).strip().lower()
                taxonomy_source = str(taxonomy_cls.get("taxonomy_source", "")).strip().lower()
                mapping_type = str(taxonomy_cls.get("mapping_type", "")).strip().lower()
                if role_norm == "code_legacy":
                    pass
                elif taxonomy_source in {
                    "codebook",
                    "codebook_specialized",
                    "description",
                    "registry_override",
                    "registry_override_specialized",
                }:
                    pass
                else:
                    issues.append(
                        "code label '{}' must be mapped via codebook/description or explicitly marked "
                        "role='code_legacy'".format(label)
                    )
                if role_norm != "code_legacy" and not mapping_type:
                    issues.append(
                        "code label '{}' mapping is missing mapping_type quality metadata".format(label)
                    )

    extra_types = [t for t in declared_damage_types if str(t).strip() and str(t) not in damage_target_families]
    if extra_types:
        issues.append(
            "project_context.damage_types contains values not backed by damage_target classes: "
            + ", ".join(sorted(set(str(t) for t in extra_types)))
        )

    return issues


def _is_negative_class_label(name):
    token = _normalized_label(name)
    negative_keys = (
        "non",
        "nocrack",
        "negative",
        "normal",
        "background",
        "bg",
        "intact",
        "healthy",
    )
    return any(key in token for key in negative_keys)


def _classification_positive_count(class_distribution):
    if not isinstance(class_distribution, dict):
        return 0

    total = 0
    for class_name, count in class_distribution.items():
        norm = _normalized_label(class_name)
        if _is_negative_class_label(class_name):
            continue
        if any(key in norm for key in ("crack", "damage", "defect", "distress", "pothole")):
            total += int(count or 0)
    return total


def _normalize_resolution_text(raw_text):
    text = str(raw_text or "").strip()
    lower = text.lower()
    if not text:
        return text

    # Metadata should describe source/original image resolution, not
    # website-serving resized/processed variants.
    if "processed" in lower or "website" in lower or "preview" in lower:
        if "source varies" in lower or "vary" in lower:
            return "Original source resolutions vary by image"
        return "Original source image resolution (website previews may be resized)"

    return text


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
    if base in {"united_states", "unitedstates"}:
        aliases.update({"us", "usa"})
    return {a for a in aliases if a}


def _build_country_lookup_from_metadata(metadata):
    geo = metadata.get("geographic_coverage", {}) or {}
    countries = geo.get("countries", []) if isinstance(geo, dict) else []
    lookup = {}
    for item in countries if isinstance(countries, list) else []:
        name = str((item or {}).get("name", "")).strip()
        if not name:
            continue
        cid = _slugify_token(name)
        if not cid:
            continue
        lookup[cid] = {
            "id": cid,
            "label": name,
            "aliases": _country_aliases(name),
        }
    return lookup


def _infer_country_id_from_split(split_name, country_lookup):
    split_slug = _slugify_token(split_name)
    if not split_slug:
        return None

    best_country = None
    best_len = -1
    for cid, meta in country_lookup.items():
        for alias in meta.get("aliases", []):
            if not alias:
                continue
            if re.search(rf"(^|_){re.escape(alias)}(_|$)", split_slug):
                if len(alias) > best_len:
                    best_country = cid
                    best_len = len(alias)
    if best_country:
        return best_country

    if len(country_lookup) == 1:
        return next(iter(country_lookup.keys()))
    return None


def _sorted_resolution_hist(hist):
    hist = hist or {}
    return dict(
        sorted(
            ((str(k), int(v)) for k, v in hist.items()),
            key=lambda kv: (
                int(kv[0].split("x")[0]) if "x" in kv[0] and kv[0].split("x")[0].isdigit() else 10**9,
                int(kv[0].split("x")[1]) if "x" in kv[0] and kv[0].split("x")[1].isdigit() else 10**9,
            ),
        )
    )


def _resolution_entries_from_hist(hist):
    return [
        {"resolution": str(k), "count": int(v)}
        for k, v in _sorted_resolution_hist(hist).items()
    ]


def _country_resolution_map_from_splits(stats_data, country_lookup):
    out = {}
    splits = (stats_data or {}).get("splits", {}) or {}
    for split_name, split_stats in splits.items():
        cid = _infer_country_id_from_split(split_name, country_lookup)
        if not cid:
            continue
        out_hist = out.setdefault(cid, defaultdict(int))
        hist = (split_stats or {}).get("image_resolution_histogram", {}) or {}
        for key, count in hist.items():
            out_hist[str(key)] += int(count)

    normalized = {}
    for cid, hist in out.items():
        normalized[cid] = _sorted_resolution_hist(hist)
    return normalized


def _normalize_metadata_for_package(package_dir):
    """
    Centralized post-patch normalization so dataset-specific patch scripts
    cannot regress core metadata semantics.
    """
    metadata_path = os.path.join(package_dir, "METADATA.json")
    if not os.path.exists(metadata_path):
        return False

    with open(metadata_path, "r", encoding="utf-8-sig") as f:
        metadata = json.load(f)

    changed = False
    project = metadata.get("project_context", {}) or {}
    stats = metadata.get("statistics", {}) or {}
    geo = metadata.get("geographic_coverage", {}) or {}
    countries = geo.get("countries", []) if isinstance(geo, dict) else []

    stats_path = os.path.join(package_dir, "stats", "stats.json")
    stats_data = {}
    if os.path.isfile(stats_path):
        try:
            with open(stats_path, "r", encoding="utf-8-sig") as f:
                stats_data = json.load(f)
        except Exception:
            stats_data = {}

    imaging = metadata.get("imaging_setup", {}) or {}
    if isinstance(imaging, dict):
        global_hist = ((stats_data or {}).get("global", {}) or {}).get("image_resolution_histogram", {}) or {}
        global_entries = _resolution_entries_from_hist(global_hist)

        if global_entries:
            global_res_list = [item["resolution"] for item in global_entries]
            if isinstance(countries, list) and len(countries) <= 1:
                res_text = ", ".join(global_res_list)
                if imaging.get("resolution") != res_text:
                    imaging["resolution"] = res_text
                    changed = True
            else:
                country_lookup = _build_country_lookup_from_metadata(metadata)
                country_hist_map = _country_resolution_map_from_splits(stats_data, country_lookup)
                by_country = []
                for cid, meta in country_lookup.items():
                    hist = country_hist_map.get(cid, {})
                    if not hist:
                        continue
                    entries = _resolution_entries_from_hist(hist)
                    by_country.append(
                        {
                            "name": meta.get("label", cid),
                            "resolutions": [e["resolution"] for e in entries],
                            "resolution_details": entries,
                        }
                    )

                if by_country:
                    if imaging.get("resolution") != "Country-dependent source resolutions":
                        imaging["resolution"] = "Country-dependent source resolutions"
                        changed = True
                    if imaging.get("resolution_by_country") != by_country:
                        imaging["resolution_by_country"] = by_country
                        changed = True
                else:
                    res_text = ", ".join(global_res_list)
                    if imaging.get("resolution") != res_text:
                        imaging["resolution"] = res_text
                        changed = True

            if imaging.get("resolution_details") != global_entries:
                imaging["resolution_details"] = global_entries
                changed = True
        elif "resolution" in imaging:
            normalized_resolution = _normalize_resolution_text(imaging.get("resolution"))
            if normalized_resolution != imaging.get("resolution"):
                imaging["resolution"] = normalized_resolution
                changed = True

        metadata["imaging_setup"] = imaging

    # For single-country classification datasets, country totals should track
    # positive/cracked samples, not all images.
    task = str(project.get("task", "") or "").strip().lower()
    if task == "classification" and isinstance(countries, list) and len(countries) == 1:
        class_distribution = stats.get("class_distribution", {}) or {}
        positive_count = _classification_positive_count(class_distribution)
        if positive_count > 0:
            country = dict(countries[0] or {})
            if int(country.get("image_count", 0) or 0) != positive_count:
                country["image_count"] = positive_count
                changed = True
            if int(country.get("annotated_image_count", 0) or 0) != positive_count:
                country["annotated_image_count"] = positive_count
                changed = True

            expected_cov = round(
                int(country.get("annotated_image_count", positive_count) or positive_count)
                / max(1, int(country.get("image_count", positive_count) or positive_count)),
                4,
            )
            if float(country.get("annotation_coverage", -1)) != expected_cov:
                country["annotation_coverage"] = expected_cov
                changed = True

            geo["countries"] = [country]
            metadata["geographic_coverage"] = geo

    # For single-country datasets, remove country drilldown artifacts.
    if isinstance(countries, list) and len(countries) <= 1:
        country_views_stats = os.path.join(package_dir, "stats", "country_views.json")
        country_views_vis = os.path.join(package_dir, "visualizations", "country_views")
        manifest_path = os.path.join(package_dir, "stats", "manifest.json")

        if os.path.isfile(country_views_stats):
            os.remove(country_views_stats)
        if os.path.isdir(country_views_vis):
            shutil.rmtree(country_views_vis, ignore_errors=True)

        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8-sig") as f:
                    manifest = json.load(f)
                artifacts = manifest.get("artifacts", []) if isinstance(manifest, dict) else []
                filtered = [
                    a for a in artifacts
                    if str((a or {}).get("name", "")) != "country_views"
                    and str((a or {}).get("file", "")) != "stats/country_views.json"
                ]
                if len(filtered) != len(artifacts):
                    manifest["artifacts"] = filtered
                    _write_json(manifest_path, manifest)
            except Exception:
                pass

    if changed:
        _write_json(metadata_path, metadata)
    return changed


def validate_dataset_package(package_dir):
    missing_files = []
    for rel in REQUIRED_FILES:
        if not os.path.exists(os.path.join(package_dir, rel)):
            missing_files.append(rel)

    metadata_path = os.path.join(package_dir, "METADATA.json")
    missing_fields = []
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            for field in REQUIRED_METADATA_FIELDS:
                val = _get_nested(metadata, field)
                if val is None or val == "" or val == []:
                    missing_fields.append(field)
            missing_fields.extend(_validate_class_taxonomy_semantics(metadata))
        except Exception as e:
            missing_fields.append(f"METADATA.json parse error: {e}")
    else:
        missing_fields.extend(REQUIRED_METADATA_FIELDS)

    is_valid = len(missing_files) == 0 and len(missing_fields) == 0
    return is_valid, missing_files, missing_fields


def print_validation_report(package_dir):
    normalized = _normalize_metadata_for_package(package_dir)
    ok, missing_files, missing_fields = validate_dataset_package(package_dir)

    if ok:
        if normalized:
            print("✓ Applied centralized metadata normalization")
        print("✓ Strict validation passed")
        return True

    print("❌ Strict validation failed")
    if missing_files:
        print("  Missing files:")
        for f in missing_files:
            print(f"   - {f}")
    if missing_fields:
        print("  Missing metadata fields:")
        for fld in missing_fields:
            print(f"   - {fld}")

    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate dataset package completeness")
    parser.add_argument("--package", required=True, help="Path to generated dataset package")
    args = parser.parse_args()

    success = print_validation_report(args.package)
    raise SystemExit(0 if success else 1)
