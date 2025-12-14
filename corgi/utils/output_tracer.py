"""
Output tracing infrastructure for comprehensive pipeline logging.

Captures raw model outputs, parsed outputs, and intermediate states
for all pipeline stages with full metadata.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.trace_types import OutputTraceEntry, StageTrace

logger = logging.getLogger(__name__)


class OutputTracer:
    """Captures all model outputs (raw, parsed, intermediate) for tracing."""

    def __init__(self, output_dir: Path, enabled: bool = True):
        """
        Initialize output tracer.

        Args:
            output_dir: Base output directory for saving traces
            enabled: Whether tracing is enabled
        """
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.entries: List[OutputTraceEntry] = []
        self.stages: List[StageTrace] = []

        # Create base directories
        if self.enabled:
            self.traces_dir = self.output_dir / "traces"
            self.traces_dir.mkdir(parents=True, exist_ok=True)

    def trace_reasoning(
        self,
        raw_output: str,
        parsed_steps: List[Any],
        model_name: str,
        prompt: str,
        model_config: Optional[Dict[str, Any]] = None,
        duration_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OutputTraceEntry:
        """
        Trace reasoning stage output.

        Args:
            raw_output: Raw model response text
            parsed_steps: Parsed reasoning steps
            model_name: Name of the model used
            prompt: Input prompt
            model_config: Model configuration
            duration_ms: Duration in milliseconds
            metadata: Additional metadata

        Returns:
            OutputTraceEntry for this trace
        """
        entry = OutputTraceEntry(
            stage="reasoning",
            model_name=model_name,
            model_config=model_config or {},
            input_prompt=prompt,
            raw_output=raw_output,
            parsed_output=parsed_steps,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

        if self.enabled:
            self.entries.append(entry)
            self._save_trace_entry(entry, "reasoning")

        return entry

    def trace_grounding(
        self,
        raw_output: str,
        bboxes: List[Any],
        model_name: str,
        statement: str,
        step_index: int,
        model_config: Optional[Dict[str, Any]] = None,
        intermediate_states: Optional[List[Dict[str, Any]]] = None,
        duration_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OutputTraceEntry:
        """
        Trace grounding stage output.

        Args:
            raw_output: Raw model response
            bboxes: Parsed bounding boxes
            model_name: Name of the model used
            statement: Input statement to ground
            step_index: Step index
            model_config: Model configuration
            intermediate_states: Intermediate processing states
            duration_ms: Duration in milliseconds
            metadata: Additional metadata

        Returns:
            OutputTraceEntry for this trace
        """
        entry = OutputTraceEntry(
            stage="grounding",
            step_index=step_index,
            model_name=model_name,
            model_config=model_config or {},
            input_prompt=statement,
            raw_output=raw_output,
            parsed_output=bboxes,
            intermediate_states=intermediate_states or [],
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

        if self.enabled:
            self.entries.append(entry)
            self._save_trace_entry(entry, f"grounding_step_{step_index}")

        return entry

    def trace_captioning(
        self,
        raw_output: str,
        description: str,
        model_name: str,
        step_index: int,
        bbox_index: int,
        bbox: tuple[float, float, float, float],
        model_config: Optional[Dict[str, Any]] = None,
        intermediate_states: Optional[List[Dict[str, Any]]] = None,
        duration_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OutputTraceEntry:
        """
        Trace captioning stage output.

        Args:
            raw_output: Raw model response
            description: Parsed caption description
            model_name: Name of the model used
            step_index: Step index
            bbox_index: Bbox index
            bbox: Bounding box coordinates
            model_config: Model configuration
            intermediate_states: Intermediate processing states
            duration_ms: Duration in milliseconds
            metadata: Additional metadata

        Returns:
            OutputTraceEntry for this trace
        """
        entry = OutputTraceEntry(
            stage="captioning",
            step_index=step_index,
            bbox_index=bbox_index,
            model_name=model_name,
            model_config=model_config or {},
            raw_output=raw_output,
            parsed_output=description,
            intermediate_states=intermediate_states or [],
            duration_ms=duration_ms,
            metadata={**(metadata or {}), "bbox": list(bbox)},
        )

        if self.enabled:
            self.entries.append(entry)
            self._save_trace_entry(
                entry, f"captioning_step_{step_index}_bbox_{bbox_index}"
            )

        return entry

    def trace_synthesis(
        self,
        raw_output: str,
        answer: str,
        key_evidence: List[Any],
        model_name: str,
        prompt: str,
        evidences: List[Any],
        model_config: Optional[Dict[str, Any]] = None,
        intermediate_states: Optional[List[Dict[str, Any]]] = None,
        duration_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OutputTraceEntry:
        """
        Trace synthesis stage output.

        Args:
            raw_output: Raw model response
            answer: Parsed answer text
            key_evidence: Parsed key evidence
            model_name: Name of the model used
            prompt: Input prompt
            evidences: Input evidences provided
            model_config: Model configuration
            intermediate_states: Intermediate processing states
            duration_ms: Duration in milliseconds
            metadata: Additional metadata

        Returns:
            OutputTraceEntry for this trace
        """
        entry = OutputTraceEntry(
            stage="synthesis",
            model_name=model_name,
            model_config=model_config or {},
            input_prompt=prompt,
            input_parameters={"evidences_count": len(evidences)},
            raw_output=raw_output,
            parsed_output={"answer": answer, "key_evidence": key_evidence},
            intermediate_states=intermediate_states or [],
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

        if self.enabled:
            self.entries.append(entry)
            self._save_trace_entry(entry, "synthesis")

        return entry

    def start_stage(
        self,
        stage_name: str,
        stage_type: str,
        step_index: Optional[int] = None,
    ) -> StageTrace:
        """
        Start tracing a new stage.

        Args:
            stage_name: Name of the stage
            stage_type: Type of stage ('reasoning', 'grounding', 'captioning', 'synthesis')
            step_index: Step index if applicable

        Returns:
            StageTrace object
        """
        stage = StageTrace(
            stage_name=stage_name,
            stage_type=stage_type,
            step_index=step_index,
        )

        if self.enabled:
            self.stages.append(stage)

        return stage

    def end_stage(
        self,
        stage: StageTrace,
        image_entries: Optional[List[Any]] = None,
        output_entries: Optional[List[OutputTraceEntry]] = None,
    ) -> None:
        """
        End tracing a stage.

        Args:
            stage: StageTrace object to complete
            image_entries: Image log entries for this stage
            output_entries: Output trace entries for this stage
        """
        if not self.enabled:
            return

        stage.end_timestamp = datetime.utcnow().isoformat() + "Z"

        if stage.start_timestamp and stage.end_timestamp:
            start = datetime.fromisoformat(stage.start_timestamp.replace("Z", "+00:00"))
            end = datetime.fromisoformat(stage.end_timestamp.replace("Z", "+00:00"))
            stage.duration_ms = (end - start).total_seconds() * 1000.0

        if image_entries:
            stage.image_entries = image_entries

        if output_entries:
            stage.output_entries = output_entries

    def _save_trace_entry(self, entry: OutputTraceEntry, filename_prefix: str) -> None:
        """Save individual trace entry to JSON file."""
        try:
            filename = f"{filename_prefix}_trace.json"
            file_path = self.traces_dir / filename

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(entry.to_dict(), f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved trace entry: {file_path}")

        except Exception as e:
            logger.error(f"Failed to save trace entry: {e}", exc_info=True)

    def get_all_entries(self) -> List[OutputTraceEntry]:
        """Get all trace entries."""
        return self.entries.copy()

    def get_all_stages(self) -> List[StageTrace]:
        """Get all stage traces."""
        return self.stages.copy()

    def save_summary(self, file_path: Optional[Path] = None) -> None:
        """Save summary of all traces to JSON."""
        if not self.enabled:
            return

        if file_path is None:
            file_path = self.output_dir / "metadata" / "pipeline_metadata.json"

        file_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "total_entries": len(self.entries),
            "total_stages": len(self.stages),
            "entries": [entry.to_dict() for entry in self.entries],
            "stages": [stage.to_dict() for stage in self.stages],
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved trace summary: {file_path}")


__all__ = ["OutputTracer"]
