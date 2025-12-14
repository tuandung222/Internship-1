from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable, List

from ..core.types import GroundedEvidence, ReasoningStep

_logger = logging.getLogger(__name__)


_JSON_FENCE_RE = re.compile(r"```(?:json)?(.*?)```", re.DOTALL | re.IGNORECASE)
_STEP_MARKER_RE = re.compile(
    r"(?im)(?:^|\n)\s*(?:step\s*(\d+)|(\d+)[\.\)])\s*[:\-]?\s*"
)
_NEEDS_VISION_RE = re.compile(
    r"needs[\s_]*vision\s*[:\-]?\s*(?P<value>true|false|yes|no|required|not required|necessary|unnecessary)",
    re.IGNORECASE,
)
_REASON_RE = re.compile(r"reason\s*[:\-]\s*(?P<value>.+)", re.IGNORECASE)
_BOX_RE = re.compile(
    r"\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]"
)

_ORDINAL_WORD_MAP = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}

_NUMBER_WORD_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

_ORDINAL_STEP_RE = re.compile(
    r"(?im)\b(?P<word>first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+step\b"
)
_WORD_STEP_RE = re.compile(
    r"(?im)\bstep\s+(?P<word>one|two|three|four|five|six|seven|eight|nine|ten)\b"
)

_META_TOKENS = {"maybe", "wait", "let's", "lets", "question", "protocol"}


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "yes", "y", "1"}:
            return True
        if lowered in {"false", "f", "no", "n", "0"}:
            return False
    return False


def _extract_json_strings(text: str) -> Iterable[str]:
    """Return candidate JSON payloads from the response text."""

    fenced = _JSON_FENCE_RE.findall(text)
    if fenced:
        for body in fenced:
            yield body.strip()
    stripped = text.strip()
    if stripped:
        yield stripped


def _load_first_json(text: str) -> Any:
    """Try to extract and parse JSON from text, with fallbacks."""
    if not text or not text.strip():
        _logger.error("Empty text provided to _load_first_json")
        raise ValueError("Empty response, cannot parse JSON.")

    _logger.debug(f"Attempting to parse JSON from text (length={len(text)})")

    last_error = None
    for candidate in _extract_json_strings(text):
        if not candidate.strip():
            continue
        try:
            result = json.loads(candidate)
            _logger.debug(f"Successfully parsed JSON using standard extraction")
            return result
        except json.JSONDecodeError as err:
            last_error = err
            _logger.debug(f"JSON parse failed for candidate: {err}")
            continue

    # Additional fallback: try to find JSON-like structures
    # Look for {...} or [...] patterns
    import re

    json_pattern = (
        r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])"
    )
    matches = re.findall(json_pattern, text, re.DOTALL)
    _logger.debug(f"Found {len(matches)} JSON-like patterns via regex")

    for match in matches:
        try:
            result = json.loads(match)
            _logger.debug(f"Successfully parsed JSON using regex fallback")
            return result
        except json.JSONDecodeError:
            continue

    # Log detailed error before raising
    _logger.error(f"Failed to parse JSON from text. First 300 chars: {text[:300]}")

    if last_error:
        raise ValueError(
            f"Unable to parse JSON from response: {last_error}"
        ) from last_error
    raise ValueError("No valid JSON found in response.")


def _trim_reasoning_text(text: str) -> str:
    lowered = text.lower()
    for anchor in ("let's draft", "draft:", "structured steps", "final reasoning"):
        pos = lowered.rfind(anchor)
        if pos != -1:
            return text[pos:]
    return text


def _clean_sentence(text: str) -> str:
    return " ".join(text.strip().split())


def _normalize_step_markers(text: str) -> str:
    """Convert ordinal step markers into numeric form (e.g., 'First step' -> 'Step 1')."""

    def replace_ordinal(match: re.Match[str]) -> str:
        word = match.group("word").lower()
        num = _ORDINAL_WORD_MAP.get(word)
        return f"Step {num}" if num is not None else match.group(0)

    def replace_word_number(match: re.Match[str]) -> str:
        word = match.group("word").lower()
        num = _NUMBER_WORD_MAP.get(word)
        return f"Step {num}" if num is not None else match.group(0)

    normalized = _ORDINAL_STEP_RE.sub(replace_ordinal, text)
    normalized = _WORD_STEP_RE.sub(replace_word_number, normalized)
    return normalized


def _extract_statement(body: str) -> str | None:
    statement_match = re.search(
        r"statement\s*[:\-]\s*(.+?)(?=\s*(?:needs\s*vision|reason\s*[:\-]|$))",
        body,
        re.IGNORECASE | re.DOTALL,
    )
    if statement_match:
        candidate = statement_match.group(1)
    else:
        # Fallback: take first sentence or line before metadata
        candidate = re.split(r"(?i)needs\s*vision|reason\s*[:\-]", body)[0]

    # Clean up the candidate
    candidate = candidate.strip().rstrip(".,;:")

    # If still empty or too short, return None
    if not candidate or len(candidate) < 5:
        return None

    return _clean_sentence(candidate)


def _extract_needs_vision(body: str) -> bool:
    match = _NEEDS_VISION_RE.search(body)
    if not match:
        return True
    token = match.group("value").strip().lower()
    if token in {"not required", "unnecessary"}:
        return False
    if token in {"required", "necessary"}:
        return True
    return _to_bool(token)


def _extract_reason(body: str) -> str | None:
    match = _REASON_RE.search(body)
    if match:
        reason = match.group("value").strip()
        reason = re.split(r"(?i)needs\s*vision", reason)[0].strip()
        reason = reason.rstrip(".")
        return reason or None
    because_match = re.search(r"because\s+(.+?)(?:\.|$)", body, re.IGNORECASE)
    if because_match:
        reason = because_match.group(1).strip().rstrip(".")
        return reason or None
    return None


def _parse_step_block(index_guess: int, body: str) -> ReasoningStep | None:
    statement = _extract_statement(body)
    if not statement:
        return None
    needs_vision = _extract_needs_vision(body)
    # Try to extract need_ocr from text (look for "ocr", "text", "read" keywords)
    need_ocr = bool(
        re.search(
            r"\b(ocr|text|read|sign|document|word|letter|character)\b",
            body,
            re.IGNORECASE,
        )
    )
    reason = _extract_reason(body)
    index = index_guess if index_guess > 0 else 1
    return ReasoningStep(
        index=index,
        statement=statement,
        needs_vision=needs_vision,
        reason=reason,
        need_ocr=need_ocr,
    )


def _parse_reasoning_from_text(
    response_text: str, max_steps: int
) -> List[ReasoningStep]:
    text = _trim_reasoning_text(response_text)
    text = _normalize_step_markers(text)
    matches = list(_STEP_MARKER_RE.finditer(text))
    if not matches:
        return []
    steps_map: dict[int, ReasoningStep] = {}
    ordering: List[int] = []
    fallback_index = 1
    for idx, marker in enumerate(matches):
        start = marker.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        raw_index = marker.group(1) or marker.group(2)
        try:
            index_guess = int(raw_index) if raw_index else fallback_index
        except (TypeError, ValueError):
            index_guess = fallback_index
        if raw_index is None:
            fallback_index += 1
        step = _parse_step_block(index_guess, body)
        if step is None:
            continue
        if step.index not in steps_map:
            ordering.append(step.index)
        steps_map[step.index] = step
        if len(ordering) >= max_steps:
            break
    return [steps_map[idx] for idx in ordering[:max_steps]]


def _looks_like_meta_statement(statement: str) -> bool:
    lowered = statement.lower()
    if any(token in lowered for token in _META_TOKENS) and "step" in lowered:
        return True
    if lowered.startswith(("maybe", "wait", "let's", "lets")):
        return True
    if len(statement) > 260 and "step" in lowered:
        return True
    return False


def _prune_steps(steps: List[ReasoningStep]) -> List[ReasoningStep]:
    filtered: List[ReasoningStep] = []
    seen_statements: set[str] = set()
    for step in steps:
        normalized = step.statement.strip().lower()
        if _looks_like_meta_statement(step.statement):
            continue
        if normalized in seen_statements:
            continue
        seen_statements.add(normalized)
        filtered.append(step)
    return filtered or steps


def _extract_description(text: str, start_index: int) -> str | None:
    boundary = max(text.rfind("\n", 0, start_index), text.rfind(".", 0, start_index))
    if boundary == -1:
        boundary = 0
    snippet = text[boundary:start_index].strip(" \n.:–-")
    if not snippet:
        return None
    return _clean_sentence(snippet)


def _parse_roi_from_text(
    response_text: str, default_step_index: int, bbox_format: str = "auto"
) -> List[GroundedEvidence]:
    evidences: List[GroundedEvidence] = []
    seen: set[tuple[float, float, float, float]] = set()
    for match in _BOX_RE.finditer(response_text):
        coords_str = match.group(0).strip("[]")
        try:
            coords = [float(part.strip()) for part in coords_str.split(",")]
        except ValueError:
            continue
        if len(coords) != 4:
            continue
        try:
            bbox = _normalize_bbox(coords, source_format=bbox_format)
        except ValueError:
            continue
        key = tuple(round(c, 4) for c in bbox)
        if key in seen:
            continue
        description = _extract_description(response_text, match.start())
        evidences.append(
            GroundedEvidence(
                step_index=default_step_index,
                bbox=bbox,
                description=description,
                confidence=None,
                raw_source={"bbox": coords, "description": description},
            )
        )
        seen.add(key)
    return evidences


def parse_structured_reasoning(
    response_text: str, max_steps: int
) -> List[ReasoningStep]:
    """Parse Qwen3-VL structured reasoning output into dataclasses."""

    try:
        payload = _load_first_json(response_text)
    except ValueError as json_error:
        steps = _parse_reasoning_from_text(response_text, max_steps=max_steps)
        if steps:
            return _prune_steps(steps)[:max_steps]
        raise json_error
    if not isinstance(payload, list):
        raise ValueError("Structured reasoning response must be a JSON list.")

    steps: List[ReasoningStep] = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue
        statement = item.get("statement") or item.get("step") or item.get("text")
        if not isinstance(statement, str):
            continue
        statement = statement.strip()
        if not statement:
            continue
        step_index = item.get("index")
        if not isinstance(step_index, int):
            step_index = idx
        needs_vision = _to_bool(item.get("needs_vision") or item.get("requires_vision"))
        need_ocr = _to_bool(item.get("need_ocr") or item.get("needs_ocr") or False)
        reason = item.get("reason") or item.get("justification")
        if isinstance(reason, str):
            reason = reason.strip() or None
        else:
            reason = None
        steps.append(
            ReasoningStep(
                index=step_index,
                statement=statement,
                needs_vision=needs_vision,
                reason=reason,
                need_ocr=need_ocr,
            )
        )
        if len(steps) >= max_steps:
            break
    steps = _prune_steps(steps)[:max_steps]
    if not steps:
        raise ValueError("No reasoning steps parsed from response.")
    return steps


def _strip_think_content(text: str) -> str:
    """
    Strip <think> tags from model output.

    Used for Thinking models that generate reasoning within <think></think> tags.

    Args:
        text: Raw model output

    Returns:
        Text with think tags removed
    """
    if not text:
        return ""
    cleaned = text
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[-1]
    cleaned = cleaned.replace("<think>", "")
    return cleaned.strip()


def _normalize_bbox(
    bbox: Any, source_format: str = "auto"
) -> tuple[float, float, float, float]:
    """
    Parse and normalize bounding box coordinates.

    Args:
        bbox: Raw bounding box coordinates (list or tuple of 4 numbers)
        source_format: Format of input coordinates
            - "auto": Auto-detect format (legacy behavior)
            - "normalized": [0, 1] range
            - "qwen": [0, 999] range (Qwen3-VL format)

    Returns:
        Normalized bbox in [0, 1] format as (x1, y1, x2, y2)
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"Bounding box must be a list of 4 numbers, got {bbox!r}")

    # Parse raw values
    coords = []
    for raw in bbox:
        if isinstance(raw, str):
            raw = raw.strip()
            if not raw:
                raw = 0
            else:
                raw = float(raw)
        elif isinstance(raw, (int, float)):
            raw = float(raw)
        else:
            raw = 0.0
        coords.append(raw)

    # Convert based on format
    if source_format == "qwen":
        # Qwen format: [0, 999] -> [0, 1]
        # Round to integers first (Qwen3-VL should output integers in [0, 999] range)
        coords = [int(round(v)) for v in coords]
        coords = [max(0.0, min(v / 999.0, 1.0)) for v in coords]
    elif source_format == "normalized":
        # Already normalized: just clamp to [0, 1]
        coords = [max(0.0, min(v, 1.0)) for v in coords]
    elif source_format == "auto":
        # Auto-detect based on magnitude (legacy behavior)
        scale = max(abs(v) for v in coords) if coords else 1.0
        if scale > 1.5:  # assume 0..1000 or pixel coordinates
            coords = [max(0.0, min(v / 1000.0, 1.0)) for v in coords]
        else:
            coords = [max(0.0, min(v, 1.0)) for v in coords]
    else:
        raise ValueError(f"Unknown source_format: {source_format}")

    # Ensure proper ordering
    x1, y1, x2, y2 = coords
    x_min, x_max = sorted((x1, x2))
    y_min, y_max = sorted((y1, y2))
    return (x_min, y_min, x_max, y_max)


# Alias for backward compatibility and clarity
_parse_bbox = _normalize_bbox


_EVIDENCE_LIST_CANDIDATE_KEYS = (
    "evidences",
    "regions",
    "boxes",
    "bboxes",
    "bbox_list",
    "detections",
    "items",
    "results",
    "data",
)


def _looks_like_evidence_dict(obj: object) -> bool:
    if not isinstance(obj, dict):
        return False
    return any(key in obj for key in ("bbox", "bbox_2d", "box"))


def _extract_evidence_sequence(payload: object) -> List[dict] | None:
    """Best-effort extraction of evidence list from arbitrary JSON payloads."""
    if isinstance(payload, list):
        if not payload or any(isinstance(item, dict) for item in payload):
            return payload
        return (
            None  # Lists of primitives (e.g., bbox arrays) are not valid evidence lists
        )

    if isinstance(payload, dict):
        # Preferred: explicit candidate keys
        for key in _EVIDENCE_LIST_CANDIDATE_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                extracted = _extract_evidence_sequence(value)
                if extracted is not None:
                    return extracted

        # Fallback: payload itself looks like an evidence dict
        if _looks_like_evidence_dict(payload):
            return [payload]

        # Secondary: nested lists/dicts
        for value in payload.values():
            if isinstance(value, dict):
                extracted = _extract_evidence_sequence(value)
                if extracted is not None:
                    return extracted
            if isinstance(value, list):
                extracted = _extract_evidence_sequence(value)
                if extracted is not None:
                    return extracted

    return None


def parse_roi_evidence(
    response_text: str, default_step_index: int, bbox_format: str = "auto"
) -> List[GroundedEvidence]:
    """
    Parse ROI grounding output into evidence structures.

    Args:
        response_text: Model response containing bbox data
        default_step_index: Default step index if not in response
        bbox_format: Format of bboxes in response
            - "auto": Auto-detect format (legacy behavior)
            - "normalized": [0, 1] range
            - "qwen": [0, 999] range (Qwen3-VL format)

    Returns:
        List of GroundedEvidence with bboxes normalized to [0, 1]
    """
    try:
        payload = _load_first_json(response_text)
    except ValueError:
        return _parse_roi_from_text(
            response_text,
            default_step_index=default_step_index,
            bbox_format=bbox_format,
        )

    payload_list = _extract_evidence_sequence(payload)
    if payload_list is None:
        # More helpful error message
        if isinstance(payload, dict):
            keys = list(payload.keys())[:5]  # Show first 5 keys
            raise ValueError(
                f"ROI extraction response must be a JSON list or dict with 'evidences' key. "
                f"Found dict with keys: {keys}. Expected format: {{'evidences': [...]}}"
            )
        elif isinstance(payload, list):
            raise ValueError(
                f"ROI extraction response must be a JSON list or dict with 'evidences' key. "
                f"Found list with {len(payload)} items, but items don't contain bbox fields."
            )
        else:
            raise ValueError(
                f"ROI extraction response must be a JSON list or dict with 'evidences' key. "
                f"Got {type(payload).__name__}: {str(payload)[:200]}"
            )

    evidences: List[GroundedEvidence] = []
    for item in payload_list:
        if not isinstance(item, dict):
            continue
        raw_bbox = item.get("bbox") or item.get("bbox_2d") or item.get("box")
        if raw_bbox is None:
            continue
        try:
            bbox = _normalize_bbox(raw_bbox, source_format=bbox_format)
        except ValueError:
            continue
        step_index = item.get("step") or item.get("step_index")
        if step_index is None:
            step_index = default_step_index
        if step_index is None or not isinstance(step_index, int):
            # Skip if no valid step_index found and no default provided
            if default_step_index is None:
                _logger.warning(f"Skipping evidence item without step_index: {item}")
                continue
            step_index = default_step_index
        description = (
            item.get("description") or item.get("caption") or item.get("detail")
        )
        if isinstance(description, str):
            description = description.strip() or None
        else:
            description = None
        confidence = (
            item.get("confidence") or item.get("score") or item.get("probability")
        )
        if isinstance(confidence, str):
            confidence = confidence.strip()
            confidence = float(confidence) if confidence else None
        elif isinstance(confidence, (int, float)):
            confidence = float(confidence)
        else:
            confidence = None
        evidences.append(
            GroundedEvidence(
                step_index=step_index,
                bbox=bbox,
                description=description,
                confidence=confidence,
                raw_source=item,
            )
        )
    return evidences


# ============================================================================
# HYBRID PARSING FOR INSTRUCT MODELS (CoT + JSON)
# ============================================================================


def _split_markdown_sections(text: str) -> dict[str, str]:
    """
    Split text by markdown headers (# Section).

    Args:
        text: Text with markdown sections

    Returns:
        Dictionary mapping section titles (lowercased) to section content
    """
    sections = {}
    current_section = None
    current_content = []

    for line in text.splitlines():
        # Check if this is a markdown header
        if line.strip().startswith("#"):
            # Save previous section if exists
            if current_section is not None:
                sections[current_section] = "\n".join(current_content).strip()

            # Start new section
            current_section = line.strip().lstrip("#").strip().rstrip(":").lower()
            current_content = []
        elif current_section is not None:
            current_content.append(line)
        else:
            # Content before any header - add to "preamble"
            if "preamble" not in sections:
                sections["preamble"] = ""
            sections["preamble"] += line + "\n"

    # Save last section
    if current_section is not None:
        sections[current_section] = "\n".join(current_content).strip()

    return sections


def _parse_json_steps(json_section: str, max_steps: int) -> List[ReasoningStep]:
    """
    Parse JSON section containing structured steps.

    Args:
        json_section: Text containing JSON with steps
        max_steps: Maximum steps to extract

    Returns:
        List of ReasoningStep objects

    Raises:
        ValueError: If JSON parsing fails
    """
    # Try to load JSON
    try:
        payload = _load_first_json(json_section)
    except Exception as e:
        _logger.error(f"Failed to parse JSON steps: {e}")
        raise ValueError(f"Invalid JSON in steps section: {e}")

    # Handle both {"steps": [...]} and direct list [...]
    if isinstance(payload, dict):
        steps_data = payload.get("steps", [])
    elif isinstance(payload, list):
        steps_data = payload
    else:
        raise ValueError(f"Expected dict or list, got {type(payload)}")

    steps = []
    for idx, item in enumerate(steps_data, start=1):
        if not isinstance(item, dict):
            continue

        statement = item.get("statement") or item.get("step") or item.get("text")
        if not isinstance(statement, str) or not statement.strip():
            continue

        step_index = item.get("index")
        if not isinstance(step_index, int):
            step_index = idx

        needs_vision = _to_bool(item.get("needs_vision") or item.get("requires_vision"))

        reason = item.get("reason") or item.get("justification")
        if isinstance(reason, str):
            reason = reason.strip() or None
        else:
            reason = None

        need_ocr = _to_bool(item.get("need_ocr") or item.get("needs_ocr") or False)
        steps.append(
            ReasoningStep(
                index=step_index,
                statement=statement.strip(),
                needs_vision=needs_vision,
                reason=reason,
                need_ocr=need_ocr,
            )
        )

        if len(steps) >= max_steps:
            break

    if not steps:
        raise ValueError("No valid steps found in JSON")

    return steps


def _regex_extract_steps(text: str, max_steps: int) -> List[ReasoningStep]:
    """
    Extract steps using regex patterns (fallback method).

    Args:
        text: Full text to parse
        max_steps: Maximum steps to extract

    Returns:
        List of ReasoningStep objects (may be empty)
    """
    # Use existing regex-based parser
    return _parse_reasoning_from_text(text, max_steps=max_steps)


def parse_hybrid_reasoning_response(
    response: str, max_steps: int, extraction_method: str = "hybrid"
) -> tuple[str, List[ReasoningStep]]:
    """
    Parse hybrid response containing CoT text + JSON structured steps.

    This is the main parser for Instruct models that output both natural language
    reasoning and structured JSON in their response.

    Strategy:
    1. Split by markdown section headers (# Reasoning, # Steps to verify)
    2. Extract JSON from fenced code block
    3. Fallback: Use regex to find JSON anywhere in response
    4. If extraction_method is "llm" or "hybrid": Call LLM to extract (not implemented yet)
    5. Final fallback: Use legacy text parser

    Args:
        response: Raw model output
        max_steps: Max steps to extract
        extraction_method: "regex", "llm", or "hybrid"

    Returns:
        Tuple of (cot_text, structured_steps) where:
        - cot_text: Natural language reasoning chain
        - structured_steps: List of ReasoningStep objects
    """
    _logger.debug(f"Parsing hybrid response with extraction_method={extraction_method}")

    # Step 1: Split response into sections
    sections = _split_markdown_sections(response)

    # Extract CoT text (typically in "reasoning" section)
    cot_text = (
        sections.get("reasoning", "")
        or sections.get("chain of thought", "")
        or sections.get("thinking", "")
        or sections.get("preamble", "")
    )

    # Extract JSON section (typically in "steps to verify" or similar)
    json_section = (
        sections.get("steps to verify", "")
        or sections.get("steps", "")
        or sections.get("verification steps", "")
        or sections.get("json", "")
    )

    # Step 2: Try to parse JSON from dedicated section
    if json_section:
        try:
            steps = _parse_json_steps(json_section, max_steps)
            _logger.debug(f"Successfully parsed {len(steps)} steps from JSON section")
            return cot_text, steps
        except Exception as e:
            _logger.warning(f"JSON parsing from dedicated section failed: {e}")

    # Step 3: Try to find and parse JSON anywhere in response
    try:
        steps = _parse_json_steps(response, max_steps)
        _logger.debug(f"Successfully parsed {len(steps)} steps from full response")
        # If we found steps but no CoT text, use the preamble or full text
        if not cot_text:
            cot_text = sections.get("preamble", "") or response[:500]
        return cot_text, steps
    except Exception as e:
        _logger.warning(f"JSON parsing from full response failed: {e}")

    # Step 4: Fallback based on extraction_method
    # TODO: Implement LLM-based extraction if needed
    # if extraction_method in ("llm", "hybrid"):
    #     try:
    #         steps = _llm_extract_steps(cot_text or response, max_steps)
    #         return cot_text or response, steps
    #     except Exception as e:
    #         _logger.warning(f"LLM extraction failed: {e}")

    # Step 5: Final fallback to regex-based parser
    _logger.info("Falling back to regex-based step extraction")
    try:
        steps = _regex_extract_steps(response, max_steps)
        if steps:
            _logger.debug(f"Regex extracted {len(steps)} steps")
            return cot_text or response, steps
    except Exception as e:
        _logger.error(f"Regex extraction also failed: {e}")

    # Last resort: return minimal response
    _logger.error("All parsing methods failed, returning fallback step")
    fallback_step = ReasoningStep(
        index=1,
        statement=cot_text[:200] if cot_text else response[:200],
        needs_vision=True,
        reason="Fallback due to parsing failure",
        need_ocr=False,
    )
    return cot_text or response, [fallback_step]
