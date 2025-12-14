"""
Utilities for resolving and validating compute devices.

Provides helpers to normalize requested device strings (e.g., "cuda:7")
to devices that are actually visible to PyTorch on the current host.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

try:  # Torch is required by the rest of the project, but keep import guarded for safety.
    import torch
except ImportError:  # pragma: no cover - torch is a runtime dependency
    torch = None  # type: ignore

logger = logging.getLogger("corgi.device_utils")


def normalize_device(device: str, *, logger_override: Optional[logging.Logger] = None) -> str:
    """
    Normalize a requested device string to one that is valid on this host.

    Args:
        device: Requested device string (e.g., "cuda:7", "cuda", "cpu").
        logger_override: Optional logger to use for warnings/info.

    Returns:
        A device string that PyTorch can use safely.
        Falls back to the last visible CUDA device or CPU when needed.

    Raises:
        ValueError: If the device string is empty.
    """
    log = logger_override or logger

    if device is None or str(device).strip() == "":
        raise ValueError("device must be specified in config (e.g., 'cuda:0' or 'cpu')")

    if torch is None:  # pragma: no cover - defensive: torch should exist in runtime env
        return device

    requested = str(device).strip()
    normalized = requested.lower()

    if normalized == "cuda":
        normalized = "cuda:0"

    if not normalized.startswith("cuda"):
        return requested  # cpu/mps/etc. - nothing to normalize

    if not torch.cuda.is_available():
        log.warning(
            "Requested CUDA device '%s' but CUDA is unavailable. Falling back to CPU.",
            requested,
        )
        return "cpu"

    try:
        device_id = int(normalized.split(":")[1])
    except (IndexError, ValueError):
        log.warning(
            "Unable to parse CUDA device from '%s'; defaulting to 'cuda:0'.",
            requested,
        )
        device_id = 0
        normalized = "cuda:0"

    num_visible = torch.cuda.device_count()
    if num_visible <= 0:
        log.warning(
            "Requested CUDA device '%s' but no CUDA devices are visible. Falling back to CPU.",
            requested,
        )
        return "cpu"

    if device_id >= num_visible:
        fallback_idx = num_visible - 1
        visible_desc = ", ".join(f"cuda:{i}" for i in range(num_visible)) or "none"
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        extra = f" (CUDA_VISIBLE_DEVICES={cuda_visible})" if cuda_visible else ""
        log.warning(
            "Requested CUDA device '%s' is not accessible%s. Falling back to 'cuda:%d' (visible devices: %s).",
            requested,
            extra,
            fallback_idx,
            visible_desc,
        )
        normalized = f"cuda:{fallback_idx}"

    return normalized


__all__ = ["normalize_device"]

