"""
Coordinate conversion utilities for bounding boxes.

Handles conversion between different coordinate formats:
- [0, 1] normalized coordinates
- [0, 999] Qwen-specific format
- Pixel coordinates
"""

from __future__ import annotations

from typing import Tuple, List, Optional


def to_qwen_format(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Convert normalized [0, 1] bbox to Qwen [0, 999] format.
    
    Args:
        bbox: Bounding box in normalized [0, 1] format (x1, y1, x2, y2)
    
    Returns:
        Bounding box in Qwen [0, 999] format
    """
    x1, y1, x2, y2 = bbox
    return (
        max(0.0, min(999.0, x1 * 1000.0)),
        max(0.0, min(999.0, y1 * 1000.0)),
        max(0.0, min(999.0, x2 * 1000.0)),
        max(0.0, min(999.0, y2 * 1000.0)),
    )


def from_qwen_format(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Convert Qwen [0, 999] bbox to normalized [0, 1] format.
    
    Args:
        bbox: Bounding box in Qwen [0, 999] format (x1, y1, x2, y2)
    
    Returns:
        Bounding box in normalized [0, 1] format
    """
    x1, y1, x2, y2 = bbox
    return (
        max(0.0, min(1.0, x1 / 1000.0)),
        max(0.0, min(1.0, y1 / 1000.0)),
        max(0.0, min(1.0, x2 / 1000.0)),
        max(0.0, min(1.0, y2 / 1000.0)),
    )


def to_pixel_format(
    bbox: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """
    Convert normalized [0, 1] bbox to pixel coordinates.
    
    Args:
        bbox: Bounding box in normalized [0, 1] format (x1, y1, x2, y2)
        image_width: Image width in pixels
        image_height: Image height in pixels
    
    Returns:
        Bounding box in pixel coordinates (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    return (
        int(max(0, min(image_width - 1, x1 * image_width))),
        int(max(0, min(image_height - 1, y1 * image_height))),
        int(max(0, min(image_width, x2 * image_width))),
        int(max(0, min(image_height, y2 * image_height))),
    )


def from_pixel_format(
    bbox: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    """
    Convert pixel coordinates to normalized [0, 1] format.
    
    Args:
        bbox: Bounding box in pixel coordinates (x1, y1, x2, y2)
        image_width: Image width in pixels
        image_height: Image height in pixels
    
    Returns:
        Bounding box in normalized [0, 1] format
    """
    x1, y1, x2, y2 = bbox
    return (
        max(0.0, min(1.0, x1 / image_width)),
        max(0.0, min(1.0, y1 / image_height)),
        max(0.0, min(1.0, x2 / image_width)),
        max(0.0, min(1.0, y2 / image_height)),
    )


def to_normalized_0_1(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Ensure bbox is in normalized [0, 1] format.
    
    Args:
        bbox: Bounding box in any format
    
    Returns:
        Bounding box in normalized [0, 1] format
    """
    x1, y1, x2, y2 = bbox
    # Auto-detect format based on magnitude
    scale = max(abs(v) for v in (x1, y1, x2, y2))
    if scale > 1.5:  # Likely [0, 999] or pixel format
        if scale > 100:  # Likely pixel format
            # Assume pixel format, but we need image dimensions
            # For now, just normalize assuming max dimension is 1000
            return (
                max(0.0, min(1.0, x1 / 1000.0)),
                max(0.0, min(1.0, y1 / 1000.0)),
                max(0.0, min(1.0, x2 / 1000.0)),
                max(0.0, min(1.0, y2 / 1000.0)),
            )
        else:  # Likely [0, 999]
            return from_qwen_format(bbox)
    else:  # Already normalized
        return (
            max(0.0, min(1.0, x1)),
            max(0.0, min(1.0, y1)),
            max(0.0, min(1.0, x2)),
            max(0.0, min(1.0, y2)),
        )


def from_normalized_0_1(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Ensure bbox is in normalized [0, 1] format (no-op, for consistency).
    
    Args:
        bbox: Bounding box in normalized [0, 1] format
    
    Returns:
        Bounding box in normalized [0, 1] format
    """
    return to_normalized_0_1(bbox)


def detect_bbox_format(bbox: Tuple[float, float, float, float]) -> str:
    """
    Detect the format of a bounding box.
    
    Args:
        bbox: Bounding box coordinates
    
    Returns:
        Format string: "normalized", "qwen", or "pixel"
    """
    x1, y1, x2, y2 = bbox
    scale = max(abs(v) for v in (x1, y1, x2, y2))
    
    if scale > 100:
        return "pixel"
    elif scale > 1.5:
        return "qwen"
    else:
        return "normalized"


def validate_bbox(bbox: Tuple[float, float, float, float], format: str = "normalized") -> bool:
    """
    Validate that a bounding box is in the correct format and has valid values.
    
    Args:
        bbox: Bounding box coordinates
        format: Expected format ("normalized", "qwen", or "pixel")
    
    Returns:
        True if valid, False otherwise
    """
    x1, y1, x2, y2 = bbox
    
    # Check that x2 > x1 and y2 > y1
    if x2 <= x1 or y2 <= y1:
        return False
    
    # Check format-specific ranges
    if format == "normalized":
        return all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2))
    elif format == "qwen":
        return all(0.0 <= v <= 999.0 for v in (x1, y1, x2, y2))
    elif format == "pixel":
        return all(v >= 0 for v in (x1, y1, x2, y2))
    else:
        return False


def calculate_iou(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float]
) -> float:
    """
    Calculate Intersection over Union (IoU) for two bboxes in normalized [0,1] format.
    
    Args:
        bbox1: First bounding box (x1, y1, x2, y2) in normalized [0, 1] format
        bbox2: Second bounding box (x1, y1, x2, y2) in normalized [0, 1] format
    
    Returns:
        IoU value in range [0, 1]
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Check if there's no intersection
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def non_maximum_suppression(
    bboxes: List[Tuple[float, float, float, float]],
    iou_threshold: float = 0.5,
    confidences: Optional[List[float]] = None
) -> List[Tuple[float, float, float, float]]:
    """
    Apply Non-Maximum Suppression (NMS) to remove overlapping bboxes.
    
    Args:
        bboxes: List of bounding boxes in normalized [0, 1] format (x1, y1, x2, y2)
        iou_threshold: IoU threshold for suppression (default: 0.5)
        confidences: Optional list of confidence scores for each bbox (higher is better)
    
    Returns:
        Filtered list of non-overlapping bboxes
    """
    if len(bboxes) <= 1:
        return bboxes
    
    # Sort by confidence (if available) or area (largest first)
    if confidences and len(confidences) == len(bboxes):
        indices = sorted(range(len(bboxes)), key=lambda i: confidences[i], reverse=True)
    else:
        # Sort by area (largest first)
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
        indices = sorted(range(len(bboxes)), key=lambda i: areas[i], reverse=True)
    
    keep = []
    suppressed = set()
    
    for i in indices:
        if i in suppressed:
            continue
        
        keep.append(bboxes[i])
        
        # Suppress overlapping bboxes
        for j in indices:
            if j == i or j in suppressed:
                continue
            
            iou = calculate_iou(bboxes[i], bboxes[j])
            if iou > iou_threshold:
                suppressed.add(j)
    
    return keep


__all__ = [
    "to_qwen_format",
    "from_qwen_format",
    "to_pixel_format",
    "from_pixel_format",
    "to_normalized_0_1",
    "from_normalized_0_1",
    "detect_bbox_format",
    "validate_bbox",
    "calculate_iou",
    "non_maximum_suppression",
]


