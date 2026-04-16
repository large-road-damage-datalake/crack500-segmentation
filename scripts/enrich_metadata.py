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
import unicodedata
from pathlib import Path
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


def _display_label(name):
    text = str(name or '').strip().replace('_', ' ').replace('-', ' ').replace('.', ' ')
    text = ' '.join(text.split())
    if not text:
        return 'Unknown'
    return ' '.join(w.capitalize() for w in text.split())


def _norm_label_key(value):
    text = unicodedata.normalize('NFKC', str(value or ''))
    text = text.strip().lower()
    text = text.replace('&', ' and ')
    text = re.sub(r'[_\-/]+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]+', ' ', text)
    return ' '.join(text.split())


def _is_code_label(name):
    token = re.sub(r'\s+', '', str(name or ''))
    return bool(re.match(r'^[A-Za-z]+[0-9][A-Za-z0-9]*$', token))


def _load_taxonomy_registry():
    registry_path = Path(__file__).with_name('label_taxonomy_registry.json')
    if not registry_path.exists():
        return {'version': 'road_damage_v1', 'canonical_classes': []}

    try:
        with registry_path.open('r', encoding='utf-8-sig') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {'version': 'road_damage_v1', 'canonical_classes': []}
        if not isinstance(data.get('canonical_classes'), list):
            data['canonical_classes'] = []
        data.setdefault('version', 'road_damage_v1')
        return data
    except Exception:
        return {'version': 'road_damage_v1', 'canonical_classes': []}


def _build_taxonomy_lookup(registry):
    alias_lookup = {}
    canonical_by_name = {}

    for item in registry.get('canonical_classes', []) or []:
        if not isinstance(item, dict):
            continue
        canonical = str(item.get('canonical_name', '')).strip()
        if not canonical:
            continue

        canonical_by_name[canonical] = item
        variants = []
        variants.extend(item.get('aliases', []) or [])
        variants.append(item.get('display_name', ''))
        variants.append(canonical)
        variants.append(canonical.replace('.', ' '))
        variants.append(canonical.replace('.', '_'))

        for variant in variants:
            key = _norm_label_key(variant)
            if key and key not in alias_lookup:
                alias_lookup[key] = canonical

    return alias_lookup, canonical_by_name


def _normalize_codebook(codebook):
    if not isinstance(codebook, dict):
        return {}

    out = {}
    for key, value in codebook.items():
        nk = _norm_label_key(key)
        if nk:
            out[nk] = value
    return out


def _description_candidates(description):
    text = str(description or '').strip()
    if not text:
        return []

    raw = unicodedata.normalize('NFKC', text)
    parts = re.split(r'[;|]', raw)
    candidates = set()

    for part in parts:
        s = str(part).strip()
        if not s:
            continue
        s = re.sub(r'^[A-Za-z0-9_\-]+\s+class\s*:\s*', '', s, flags=re.IGNORECASE)
        candidates.add(_norm_label_key(s))

        core = re.sub(r'\(.*?\)', '', s).strip()
        if core:
            candidates.add(_norm_label_key(core))

        for sub in re.split(r'[,/]+', s):
            sub = sub.strip()
            if sub:
                candidates.add(_norm_label_key(sub))

    return [c for c in candidates if c]


def _lookup_registry_match(label, alias_lookup):
    key = _norm_label_key(label)
    if not key:
        return None
    return alias_lookup.get(key)


def _resolve_from_description(description, alias_lookup):
    matches = set()
    for cand in _description_candidates(description):
        canonical = alias_lookup.get(cand)
        if canonical:
            matches.add(canonical)

    if len(matches) == 1:
        return next(iter(matches))
    return None


def _resolve_codebook_mapping(label, codebook_norm):
    key = _norm_label_key(label)
    if not key:
        return None
    return codebook_norm.get(key)


def _build_resolved_entry(canonical_name, canonical_info, taxonomy_source):
    canonical_info = canonical_info or {}
    display_name = str(canonical_info.get('display_name') or _display_label(canonical_name))
    damage_family = str(canonical_info.get('damage_family') or 'unknown')
    role = str(canonical_info.get('role') or 'unknown')

    aliases = [str(a).strip() for a in (canonical_info.get('aliases') or []) if str(a).strip()]
    subtype = canonical_info.get('damage_subtype')
    qualifiers = canonical_info.get('qualifiers', [])
    if not isinstance(qualifiers, list):
        qualifiers = []

    return {
        'canonical_name': canonical_name,
        'display_name': display_name,
        'damage_family': damage_family,
        'damage_subtype': str(subtype).strip() if subtype is not None else '',
        'role': role,
        'aliases': aliases,
        'qualifiers': [str(q).strip() for q in qualifiers if str(q).strip()],
        'taxonomy_source': taxonomy_source,
        'taxonomy_unresolved': role in {'unknown', 'code_legacy'},
    }


def _resolve_taxonomy_for_label(name, description, codebook_norm, alias_lookup, canonical_by_name):
    source_name = str(name or '').strip()

    codebook_value = _resolve_codebook_mapping(source_name, codebook_norm)
    if codebook_value is not None:
        if isinstance(codebook_value, str):
            canonical_name = codebook_value.strip()
            info = canonical_by_name.get(canonical_name, {})
            if not info:
                info = {
                    'display_name': _display_label(canonical_name),
                    'damage_family': 'unknown',
                    'role': 'unknown',
                    'aliases': [source_name],
                }
            return _build_resolved_entry(canonical_name, info, 'codebook')

        if isinstance(codebook_value, dict):
            canonical_name = str(codebook_value.get('canonical_name', '')).strip()
            base = dict(canonical_by_name.get(canonical_name, {})) if canonical_name else {}
            merged = {**base, **codebook_value}
            canonical = canonical_name or str(merged.get('canonical_name', '')).strip()
            if not canonical:
                canonical = f"unknown.{_slugify_token(source_name) or 'label'}"
            merged.setdefault('display_name', _display_label(source_name))
            merged.setdefault('aliases', [source_name])
            return _build_resolved_entry(canonical, merged, 'codebook')

    canonical_name = _lookup_registry_match(source_name, alias_lookup)
    if canonical_name:
        return _build_resolved_entry(canonical_name, canonical_by_name.get(canonical_name, {}), 'registry')

    if _is_code_label(source_name):
        desc_match = _resolve_from_description(description, alias_lookup)
        if desc_match:
            return _build_resolved_entry(desc_match, canonical_by_name.get(desc_match, {}), 'description')

        fallback_code = _slugify_token(source_name)
        return {
            'canonical_name': f"code.{fallback_code or 'unknown'}",
            'display_name': _display_label(source_name),
            'damage_family': 'unknown',
            'damage_subtype': '',
            'role': 'code_legacy',
            'aliases': [source_name],
            'qualifiers': [],
            'taxonomy_source': 'fallback_code',
            'taxonomy_unresolved': True,
        }

    return {
        'canonical_name': f"unknown.{_slugify_token(source_name) or 'label'}",
        'display_name': _display_label(source_name),
        'damage_family': 'unknown',
        'damage_subtype': '',
        'role': 'unknown',
        'aliases': [source_name],
        'qualifiers': [],
        'taxonomy_source': 'fallback_unknown',
        'taxonomy_unresolved': True,
    }


def _merge_aliases(*alias_sources):
    out = []
    seen = set()
    for source in alias_sources:
        vals = source if isinstance(source, list) else [source]
        for v in vals:
            s = str(v).strip()
            if not s:
                continue
            key = _norm_label_key(s)
            if key and key not in seen:
                seen.add(key)
                out.append(s)
    return out


def _extract_class_records(metadata, class_distribution):
    records = []
    seen = set()

    schemas = metadata.get('annotation_schema', [])
    if isinstance(schemas, list):
        for schema in schemas:
            classes = schema.get('classes', []) if isinstance(schema, dict) else []
            if not isinstance(classes, list):
                continue
            for cls in classes:
                if not isinstance(cls, dict):
                    continue
                name = str(cls.get('name', '')).strip()
                if not name or name in seen:
                    continue
                seen.add(name)
                records.append(
                    {
                        'name': name,
                        'entry': cls,
                        'description': str(cls.get('description', '')).strip(),
                    }
                )

    for cls_name in (class_distribution or {}).keys():
        name = str(cls_name).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        records.append(
            {
                'name': name,
                'entry': {},
                'description': '',
            }
        )

    return records


def _normalize_class_taxonomy(metadata, class_distribution, config):
    records = _extract_class_records(metadata, class_distribution)
    if not records:
        metadata['class_taxonomy'] = {
            'version': 'road_damage_v1',
            'classes': [],
            'damage_types': [],
            'unresolved_labels': [],
        }
        return []

    count_by_name = {str(k): int(v) for k, v in (class_distribution or {}).items()}
    schemas = metadata.get('annotation_schema')
    if not isinstance(schemas, list):
        schemas = []
        metadata['annotation_schema'] = schemas

    primary_schema = None
    for schema in schemas:
        if isinstance(schema, dict) and isinstance(schema.get('classes'), list):
            primary_schema = schema
            break
    if primary_schema is None:
        primary_schema = {
            'type': 'label',
            'description': 'Auto-generated class schema from observed labels.',
            'classes': [],
        }
        schemas.append(primary_schema)

    existing_classes = primary_schema.get('classes', [])
    if not isinstance(existing_classes, list):
        existing_classes = []
    existing_by_name = {}
    used_ids = set()
    for entry in existing_classes:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get('name', '')).strip()
        if name and name not in existing_by_name:
            existing_by_name[name] = entry
        try:
            used_ids.add(int(entry.get('id')))
        except Exception:
            pass

    registry = _load_taxonomy_registry()
    alias_lookup, canonical_by_name = _build_taxonomy_lookup(registry)
    codebook_norm = _normalize_codebook(config.get('label_codebook', {}) if isinstance(config, dict) else {})

    normalized = []
    taxonomy_records = []
    next_id = 1
    unresolved_labels = []

    for record in records:
        name = record['name']
        while next_id in used_ids:
            next_id += 1

        src = existing_by_name.get(name, {})
        fallback_id = src.get('id', next_id)
        try:
            class_id = int(fallback_id)
        except Exception:
            class_id = next_id

        resolved = _resolve_taxonomy_for_label(
            name=name,
            description=record.get('description') or src.get('description') or '',
            codebook_norm=codebook_norm,
            alias_lookup=alias_lookup,
            canonical_by_name=canonical_by_name,
        )

        entry = dict(src)
        for field in (
            'display_name',
            'damage_family',
            'damage_type',
            'damage_subtype',
            'qualifiers',
            'role',
            'is_damage_target',
            'taxonomy_source',
            'taxonomy_unresolved',
            'aliases',
        ):
            entry.pop(field, None)

        canonical_name = str(resolved.get('canonical_name') or src.get('canonical_name') or _slugify_token(name))
        display_name = str(resolved.get('display_name') or src.get('display_name') or _display_label(name))
        damage_family = str(
            resolved.get('damage_family')
            or src.get('damage_family')
            or src.get('damage_type')
            or 'unknown'
        )
        damage_subtype = str(resolved.get('damage_subtype') or src.get('damage_subtype') or '').strip()
        role = str(resolved.get('role') or src.get('role') or 'unknown')
        taxonomy_source = str(resolved.get('taxonomy_source') or src.get('taxonomy_source') or 'fallback_unknown')
        taxonomy_unresolved = bool(resolved.get('taxonomy_unresolved', src.get('taxonomy_unresolved', False)))
        aliases = _merge_aliases(src.get('aliases', []), resolved.get('aliases', []), [name])

        entry['id'] = class_id
        entry['name'] = name
        entry['canonical_name'] = canonical_name

        if record.get('description') and not entry.get('description'):
            entry['description'] = record.get('description')

        if name in count_by_name:
            entry['instances'] = int(count_by_name.get(name, 0))

        if taxonomy_unresolved:
            unresolved_labels.append(name)

        taxonomy_record = {
            'canonical_name': canonical_name,
            'display_name': display_name,
            'damage_family': damage_family,
            'role': role,
            'is_damage_target': bool(role == 'damage_target'),
            'aliases': aliases,
            'source_label': name,
            'taxonomy_source': taxonomy_source,
            'taxonomy_unresolved': taxonomy_unresolved,
        }
        if damage_subtype:
            taxonomy_record['damage_subtype'] = damage_subtype
        taxonomy_records.append(taxonomy_record)

        normalized.append(entry)
        used_ids.add(class_id)
        next_id += 1

    primary_schema['classes'] = normalized

    damage_types = []
    taxonomy_agg = {}
    for cls in taxonomy_records:
        role = str(cls.get('role', '')).strip().lower()
        fam = str(cls.get('damage_family', '')).strip()
        if role == 'damage_target' and fam and fam not in damage_types and fam != 'unknown':
            damage_types.append(fam)

        canonical = str(cls.get('canonical_name', '')).strip()
        if not canonical:
            continue
        agg = taxonomy_agg.setdefault(
            canonical,
            {
                'canonical_name': canonical,
                'display_name': cls.get('display_name'),
                'damage_family': fam,
                'damage_subtype': cls.get('damage_subtype', ''),
                'role': cls.get('role'),
                'is_damage_target': bool(cls.get('is_damage_target')),
                'aliases': [],
                'source_labels': [],
                'taxonomy_source': cls.get('taxonomy_source', ''),
                'taxonomy_unresolved': bool(cls.get('taxonomy_unresolved', False)),
            },
        )
        if not agg.get('display_name') and cls.get('display_name'):
            agg['display_name'] = cls.get('display_name')
        if not agg.get('damage_family') and cls.get('damage_family'):
            agg['damage_family'] = cls.get('damage_family')
        if not agg.get('damage_subtype') and cls.get('damage_subtype'):
            agg['damage_subtype'] = cls.get('damage_subtype')
        if not agg.get('role') and cls.get('role'):
            agg['role'] = cls.get('role')
        if not agg.get('taxonomy_source') and cls.get('taxonomy_source'):
            agg['taxonomy_source'] = cls.get('taxonomy_source')
        agg['aliases'] = _merge_aliases(agg.get('aliases', []), cls.get('aliases', []))
        agg['source_labels'] = _merge_aliases(agg.get('source_labels', []), [cls.get('source_label', '')])
        agg['taxonomy_unresolved'] = bool(agg.get('taxonomy_unresolved') or cls.get('taxonomy_unresolved'))

    metadata['class_taxonomy'] = {
        'version': str(registry.get('version', 'road_damage_v1')),
        'classes': [taxonomy_agg[k] for k in sorted(taxonomy_agg.keys())],
        'damage_types': damage_types,
        'unresolved_labels': sorted(set(unresolved_labels)),
    }
    return damage_types


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
    damage_types = _normalize_class_taxonomy(
        metadata,
        metadata['statistics'].get('class_distribution', {}),
        config,
    )
    task_value = config.get('task_type', project_context.get('task', 'object_detection'))
    project_context['task'] = task_value
    # Keep task list aligned with current package task and avoid stale template defaults.
    project_context['tasks'] = [task_value]
    if damage_types:
        project_context['damage_types'] = damage_types
    elif not isinstance(project_context.get('damage_types'), list):
        project_context['damage_types'] = []
    project_context['label_taxonomy_version'] = str(
        metadata.get('class_taxonomy', {}).get('version', 'road_damage_v1')
    )

    # Keep publication/source descriptors at metadata root only.
    for dup_field in ('source_publication', 'source_dataset'):
        if dup_field in project_context:
            if dup_field not in metadata:
                metadata[dup_field] = project_context.get(dup_field)
            project_context.pop(dup_field, None)

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
    
    print("OK Enriched METADATA.json")
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
