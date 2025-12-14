"""
CoRGI Utilities Package.

This package provides utility functions for the CoRGI pipeline.
"""

from .inference_helpers import (
    # Directory setup
    setup_output_dir,
    # Font
    load_font,
    # Image annotation
    annotate_image_with_bboxes,
    DEFAULT_COLORS,
    STEP_COLORS,
    # Evidence crops
    save_evidence_crops,
    # Result saving
    save_results_json,
    save_summary_report,
    # Conversion helpers
    pipeline_result_to_dict,
    evidence_to_bbox_list,
)

from .warm_up import (
    WarmUpConfig,
    warm_up_pipeline,
    warm_up_models_only,
    verify_cuda_ready,
    create_dummy_image,
)

__all__ = [
    # inference_helpers
    "setup_output_dir",
    "load_font",
    "annotate_image_with_bboxes",
    "DEFAULT_COLORS",
    "STEP_COLORS",
    "save_evidence_crops",
    "save_results_json",
    "save_summary_report",
    "pipeline_result_to_dict",
    "evidence_to_bbox_list",
    # warm_up
    "WarmUpConfig",
    "warm_up_pipeline",
    "warm_up_models_only",
    "verify_cuda_ready",
    "create_dummy_image",
]
