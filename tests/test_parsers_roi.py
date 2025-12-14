"""
Tests for ROI evidence parsing helpers.

These tests focus on the flexible JSON formats produced by Qwen
grounding responses, ensuring we can handle variations observed in
production (e.g., missing 'evidences' key, nested fields, single-object payloads).
"""

from __future__ import annotations

import json

import pytest

from corgi.utils.parsers import parse_roi_evidence


def test_parse_roi_evidence_handles_regions_key():
    """Payloads with a 'regions' list should be parsed like 'evidences'."""
    response = json.dumps(
        {
            "regions": [
                {
                    "step": 2,
                    "bbox": [0, 0, 999, 999],
                    "description": "full frame",
                    "confidence": 0.73,
                }
            ],
            "meta": {"prompt": "ignored"},
        }
    )

    evidences = parse_roi_evidence(response, default_step_index=1, bbox_format="qwen")

    assert len(evidences) == 1
    evidence = evidences[0]
    assert evidence.step_index == 2
    assert evidence.description == "full frame"
    assert evidence.confidence == pytest.approx(0.73)
    # 0..999 coordinates should normalize to 0..1
    assert evidence.bbox == pytest.approx((0.0, 0.0, 1.0, 1.0))


def test_parse_roi_evidence_handles_nested_results_key():
    """Nested dicts containing candidate lists should also be parsed."""
    response = json.dumps(
        {
            "result": {
                "detections": [
                    {
                        "bbox": [100, 200, 800, 900],
                        "step_index": 3,
                        "description": "object near center",
                    }
                ]
            }
        }
    )

    evidences = parse_roi_evidence(response, default_step_index=1, bbox_format="qwen")

    assert len(evidences) == 1
    evidence = evidences[0]
    assert evidence.step_index == 3
    assert evidence.description == "object near center"
    assert evidence.bbox == pytest.approx(
        (100 / 999.0, 200 / 999.0, 800 / 999.0, 900 / 999.0)
    )


def test_parse_roi_evidence_handles_single_dict_payload():
    """Single dict payloads that look like evidence should be wrapped automatically."""
    response = json.dumps(
        {
            "bbox": [10, 20, 500, 600],
            "description": "single object",
        }
    )

    evidences = parse_roi_evidence(response, default_step_index=5, bbox_format="qwen")

    assert len(evidences) == 1
    evidence = evidences[0]
    assert evidence.step_index == 5  # falls back to default when not provided
    assert evidence.description == "single object"
    assert evidence.bbox == pytest.approx(
        (10 / 999.0, 20 / 999.0, 500 / 999.0, 600 / 999.0)
    )
