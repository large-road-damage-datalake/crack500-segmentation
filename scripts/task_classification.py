#!/usr/bin/env python3
"""
Task classification helpers.

Provides a normalized task classification payload, including segmentation subtype
for segmentation datasets (semantic / instance / panoptic).
"""


SEGMENTATION_LABELS = {
    "semantic_segmentation": "semantic",
    "instance_segmentation": "instance",
    "panoptic_segmentation": "panoptic",
}


def _norm_token(value):
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_segmentation_type(value):
    token = _norm_token(value)
    if not token:
        return ""

    aliases = {
        "semantic": "semantic_segmentation",
        "semantic_seg": "semantic_segmentation",
        "semantic_segmentation": "semantic_segmentation",
        "instance": "instance_segmentation",
        "instance_seg": "instance_segmentation",
        "instance_segmentation": "instance_segmentation",
        "panoptic": "panoptic_segmentation",
        "panoptic_seg": "panoptic_segmentation",
        "panoptic_segmentation": "panoptic_segmentation",
    }
    return aliases.get(token, "")


def _infer_segmentation_type(config):
    fmt = _norm_token(config.get("format", ""))
    if fmt == "png_masks":
        return "semantic_segmentation"

    if fmt == "coco":
        # If explicitly marked as panoptic-style COCO, classify as panoptic.
        panoptic_hints = [
            config.get("panoptic"),
            config.get("panoptic_format"),
            config.get("panoptic_annotations"),
            config.get("panoptic_json"),
            config.get("annotation_mode"),
            config.get("segmentation_mode"),
        ]
        merged = " ".join([_norm_token(v) for v in panoptic_hints if v is not None])
        if "panoptic" in merged:
            return "panoptic_segmentation"
        return "instance_segmentation"

    # Conservative default for segmentation datasets when unknown.
    return "semantic_segmentation"


def resolve_task_classification(config):
    """
    Return normalized task classification payload.
    """
    task_type = _norm_token(config.get("task_type", "")) or "unknown"
    payload = {
        "primary_task": task_type,
    }

    if task_type != "segmentation":
        return payload

    explicit = (
        _normalize_segmentation_type(config.get("segmentation_type"))
        or _normalize_segmentation_type(config.get("segmentation_task"))
        or _normalize_segmentation_type(config.get("task_subtype"))
    )
    seg_type = explicit or _infer_segmentation_type(config)

    payload["task_subtype"] = seg_type
    payload["segmentation_type"] = seg_type
    payload["segmentation_family"] = SEGMENTATION_LABELS.get(seg_type, "semantic")
    payload["segmentation_type_inferred"] = not bool(explicit)
    return payload
