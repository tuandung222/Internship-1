"""
Markdown Report Generator for comprehensive pipeline tracing.

Generates detailed Markdown reports with base64-embedded images,
collapsible sections using HTML details tags, and complete trace data.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from io import BytesIO

from PIL import Image

from ..core.trace_types import PipelineTrace, ImageLogEntry, OutputTraceEntry


def _image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 data URI."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


class MarkdownReportGenerator:
    """Generates comprehensive Markdown reports from trace data."""

    def __init__(self, output_dir: Path):
        """
        Initialize Markdown report generator.

        Args:
            output_dir: Output directory for the report
        """
        self.output_dir = Path(output_dir)

    def generate_report(
        self,
        trace_data: PipelineTrace,
        image_logger,
        output_tracer,
        result: Optional[Any] = None,
    ) -> Path:
        """
        Generate comprehensive Markdown report.

        Args:
            trace_data: Complete pipeline trace data
            image_logger: ImageLogger instance with all image entries
            output_tracer: OutputTracer instance with all trace entries
            result: Optional PipelineResult for additional data

        Returns:
            Path to generated Markdown file
        """
        markdown_content = self._generate_markdown(
            trace_data,
            image_logger.get_all_entries() if image_logger else [],
            output_tracer.get_all_entries() if output_tracer else [],
            output_tracer.get_all_stages() if output_tracer else [],
            result,
        )

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"trace_report_{timestamp}.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        return report_path

    def _generate_markdown(
        self,
        trace_data: PipelineTrace,
        image_entries: List[ImageLogEntry],
        output_entries: List[OutputTraceEntry],
        stage_traces: List[Any],
        result: Optional[Any] = None,
    ) -> str:
        """Generate complete Markdown content."""
        parts = [
            self._generate_header(trace_data, result),
            self._generate_summary_section(trace_data, result),
            self._generate_reasoning_section(output_entries, image_entries, result),
            self._generate_grounding_section(output_entries, image_entries),
            self._generate_captioning_section(output_entries, image_entries),
            self._generate_synthesis_section(output_entries, image_entries, result),
            self._generate_images_section(image_entries),
            self._generate_timing_section(stage_traces),
            self._generate_footer(),
        ]
        return "\n\n".join(parts)

    def _generate_header(
        self, trace_data: PipelineTrace, result: Optional[Any] = None
    ) -> str:
        """Generate Markdown header."""
        lines = [
            "# CoRGI Pipeline Trace Report",
            "",
            f"**Pipeline ID:** {trace_data.pipeline_id}",
            f"**Question:** {trace_data.question}",
            f"**Start Time:** {trace_data.start_timestamp}",
            f"**End Time:** {trace_data.end_timestamp}",
            f"**Total Duration:** {trace_data.total_duration_ms:.2f} ms",
            "",
        ]

        if result:
            lines.extend(
                [
                    f"**Final Answer:** {result.answer or '(no answer)'}",
                    f"**Reasoning Steps:** {len(result.steps) if result.steps else 0}",
                    f"**Evidence Items:** {len(result.evidence) if result.evidence else 0}",
                    f"**Key Evidence Items:** {len(result.key_evidence) if result.key_evidence else 0}",
                    "",
                ]
            )

        return "\n".join(lines)

    def _generate_summary_section(
        self, trace_data: PipelineTrace, result: Optional[Any] = None
    ) -> str:
        """Generate summary section."""
        lines = [
            "## Summary",
            "",
            "### Configuration",
            "",
        ]

        config = trace_data.config
        if isinstance(config, dict):
            lines.append("```json")
            lines.append(json.dumps(config, indent=2))
            lines.append("```")
            lines.append("")

        if result and result.cot_text:
            lines.extend(
                [
                    "### Chain of Thought (Full Text)",
                    "",
                    "```text",
                    result.cot_text,
                    "```",
                    "",
                ]
            )

        return "\n".join(lines)

    def _generate_reasoning_section(
        self,
        output_entries: List[OutputTraceEntry],
        image_entries: List[ImageLogEntry],
        result: Optional[Any] = None,
    ) -> str:
        """Generate reasoning stage section."""
        lines = [
            "## Reasoning Stage",
            "",
        ]

        # Find reasoning entries
        reasoning_entries = [e for e in output_entries if e.stage == "reasoning"]
        reasoning_images = [e for e in image_entries if e.stage == "reasoning"]

        if reasoning_entries:
            entry = reasoning_entries[0]
            lines.extend(
                [
                    "### Model Output",
                    "",
                    f"**Model:** {entry.model_name}",
                    f"**Duration:** {entry.duration_ms:.2f} ms",
                    "",
                    "<details>",
                    "<summary>Input Prompt</summary>",
                    "",
                    "```text",
                    entry.input_prompt or "N/A",
                    "```",
                    "",
                    "</details>",
                    "",
                    "<details>",
                    "<summary>Raw Output</summary>",
                    "",
                    "```text",
                    entry.raw_output or "N/A",
                    "```",
                    "",
                    "</details>",
                    "",
                ]
            )

            if entry.parsed_output:
                lines.extend(
                    [
                        "<details>",
                        "<summary>Parsed Output</summary>",
                        "",
                        "```json",
                        json.dumps(entry.parsed_output, indent=2, ensure_ascii=False),
                        "```",
                        "",
                        "</details>",
                        "",
                    ]
                )

        # Show structured steps if available
        if result and result.steps:
            lines.extend(
                [
                    "### Structured Reasoning Steps",
                    "",
                ]
            )
            for step in result.steps:
                lines.extend(
                    [
                        f"#### Step {step.index}: {step.statement}",
                        "",
                        f"- **Needs Vision:** {'Yes' if step.needs_vision else 'No'}",
                    ]
                )
                if step.reason:
                    lines.append(f"- **Reason:** {step.reason}")
                lines.append("")

        # Show reasoning images
        if reasoning_images:
            lines.extend(
                [
                    "### Reasoning Images",
                    "",
                ]
            )
            for img_entry in reasoning_images:
                img_path = self.output_dir / img_entry.file_path
                if img_path.exists():
                    try:
                        img = Image.open(img_path)
                        img_base64 = _image_to_base64(img)
                        lines.extend(
                            [
                                f"#### {img_entry.image_type} (Step {img_entry.step_index or 'N/A'})",
                                "",
                                f"![{img_entry.image_type}]({img_base64})",
                                "",
                                f"- **Size:** {img_entry.original_size}",
                                f"- **Timestamp:** {img_entry.timestamp}",
                                "",
                            ]
                        )
                    except Exception as e:
                        lines.append(f"*Error loading image: {e}*")

        return "\n".join(lines)

    def _generate_grounding_section(
        self,
        output_entries: List[OutputTraceEntry],
        image_entries: List[ImageLogEntry],
    ) -> str:
        """Generate grounding stage section."""
        lines = [
            "## Grounding Stage",
            "",
        ]

        grounding_entries = [e for e in output_entries if e.stage == "grounding"]
        grounding_images = [e for e in image_entries if e.stage == "grounding"]

        # Group by step
        by_step: Dict[Optional[int], List[OutputTraceEntry]] = {}
        for entry in grounding_entries:
            step = entry.step_index
            if step not in by_step:
                by_step[step] = []
            by_step[step].append(entry)

        for step_idx, entries in sorted(by_step.items()):
            lines.extend(
                [
                    f"### Step {step_idx}",
                    "",
                ]
            )

            for entry in entries:
                lines.extend(
                    [
                        f"**Model:** {entry.model_name}",
                        f"**Duration:** {entry.duration_ms:.2f} ms",
                        "",
                        "<details>",
                        "<summary>Input Statement</summary>",
                        "",
                        "```text",
                        entry.metadata.get("statement", "N/A"),
                        "```",
                        "",
                        "</details>",
                        "",
                        "<details>",
                        "<summary>Raw Output</summary>",
                        "",
                        "```text",
                        entry.raw_output or "N/A",
                        "```",
                        "",
                        "</details>",
                        "",
                    ]
                )

                if entry.parsed_output:
                    lines.extend(
                        [
                            "<details>",
                            "<summary>Parsed Bounding Boxes</summary>",
                            "",
                            "```json",
                            json.dumps(
                                entry.parsed_output, indent=2, ensure_ascii=False
                            ),
                            "```",
                            "",
                            "</details>",
                            "",
                        ]
                    )

            # Show images for this step
            step_images = [e for e in grounding_images if e.step_index == step_idx]
            if step_images:
                lines.append("#### Evidence Images")
                lines.append("")
                for img_entry in step_images:
                    img_path = self.output_dir / img_entry.file_path
                    if img_path.exists():
                        try:
                            img = Image.open(img_path)
                            img_base64 = _image_to_base64(img)
                            lines.extend(
                                [
                                    f"**{img_entry.image_type}** (Bbox {img_entry.bbox_index or 'N/A'})",
                                    "",
                                    f"![{img_entry.image_type}]({img_base64})",
                                    "",
                                ]
                            )
                            if img_entry.bbox:
                                lines.append(f"- **BBox:** {img_entry.bbox}")
                            lines.append("")
                        except Exception as e:
                            lines.append(f"*Error loading image: {e}*")

        return "\n".join(lines)

    def _generate_captioning_section(
        self,
        output_entries: List[OutputTraceEntry],
        image_entries: List[ImageLogEntry],
    ) -> str:
        """Generate captioning stage section."""
        lines = [
            "## Captioning Stage",
            "",
        ]

        captioning_entries = [e for e in output_entries if e.stage == "captioning"]
        captioning_images = [e for e in image_entries if e.stage == "captioning"]

        # Group by step and bbox
        by_step_bbox: Dict[
            Tuple[Optional[int], Optional[int]], List[OutputTraceEntry]
        ] = {}
        for entry in captioning_entries:
            key = (entry.step_index, entry.bbox_index)
            if key not in by_step_bbox:
                by_step_bbox[key] = []
            by_step_bbox[key].append(entry)

        for (step_idx, bbox_idx), entries in sorted(by_step_bbox.items()):
            lines.extend(
                [
                    f"### Step {step_idx} - Bbox {bbox_idx}",
                    "",
                ]
            )

            for entry in entries:
                lines.extend(
                    [
                        f"**Model:** {entry.model_name}",
                        f"**Duration:** {entry.duration_ms:.2f} ms",
                        "",
                        "<details>",
                        "<summary>Raw Output</summary>",
                        "",
                        "```text",
                        entry.raw_output or "N/A",
                        "```",
                        "",
                        "</details>",
                        "",
                    ]
                )

                if entry.parsed_output:
                    lines.extend(
                        [
                            "<details>",
                            "<summary>Parsed Caption</summary>",
                            "",
                            "```text",
                            str(entry.parsed_output),
                            "```",
                            "",
                            "</details>",
                            "",
                        ]
                    )

            # Show cropped images for this step/bbox
            step_bbox_images = [
                e
                for e in captioning_images
                if e.step_index == step_idx and e.bbox_index == bbox_idx
            ]
            if step_bbox_images:
                lines.append("#### Cropped Evidence Images")
                lines.append("")
                for img_entry in step_bbox_images:
                    img_path = self.output_dir / img_entry.file_path
                    if img_path.exists():
                        try:
                            img = Image.open(img_path)
                            img_base64 = _image_to_base64(img)
                            lines.extend(
                                [
                                    f"**{img_entry.image_type}**",
                                    "",
                                    f"![{img_entry.image_type}]({img_base64})",
                                    "",
                                ]
                            )
                            if img_entry.cropped_size:
                                lines.append(
                                    f"- **Cropped Size:** {img_entry.cropped_size}"
                                )
                            if img_entry.crop_ratio:
                                lines.append(
                                    f"- **Crop Ratio:** {img_entry.crop_ratio:.2%}"
                                )
                            lines.append("")
                        except Exception as e:
                            lines.append(f"*Error loading image: {e}*")

        return "\n".join(lines)

    def _generate_synthesis_section(
        self,
        output_entries: List[OutputTraceEntry],
        image_entries: List[ImageLogEntry],
        result: Optional[Any] = None,
    ) -> str:
        """Generate synthesis stage section."""
        lines = [
            "## Answer Synthesis Stage",
            "",
        ]

        synthesis_entries = [e for e in output_entries if e.stage == "synthesis"]
        synthesis_images = [e for e in image_entries if e.stage == "synthesis"]

        if synthesis_entries:
            entry = synthesis_entries[0]
            lines.extend(
                [
                    "### Model Output",
                    "",
                    f"**Model:** {entry.model_name}",
                    f"**Duration:** {entry.duration_ms:.2f} ms",
                    "",
                    "<details>",
                    "<summary>Input Prompt</summary>",
                    "",
                    "```text",
                    entry.input_prompt or "N/A",
                    "```",
                    "",
                    "</details>",
                    "",
                    "<details>",
                    "<summary>Raw Output</summary>",
                    "",
                    "```text",
                    entry.raw_output or "N/A",
                    "```",
                    "",
                    "</details>",
                    "",
                ]
            )

            if entry.parsed_output:
                lines.extend(
                    [
                        "<details>",
                        "<summary>Parsed Output</summary>",
                        "",
                        "```json",
                        json.dumps(entry.parsed_output, indent=2, ensure_ascii=False),
                        "```",
                        "",
                        "</details>",
                        "",
                    ]
                )

        if result:
            lines.extend(
                [
                    "### Final Answer",
                    "",
                    f"{result.answer or '(no answer)'}",
                    "",
                ]
            )

            if result.key_evidence:
                lines.extend(
                    [
                        "### Key Evidence",
                        "",
                    ]
                )
                for idx, kev in enumerate(result.key_evidence, 1):
                    lines.extend(
                        [
                            f"#### Key Evidence {idx}",
                            "",
                            f"**Description:** {kev.description}",
                            f"**Reasoning:** {kev.reasoning}",
                            f"**BBox:** {kev.bbox}",
                            "",
                        ]
                    )

        # Show synthesis images
        if synthesis_images:
            lines.extend(
                [
                    "### Synthesis Images",
                    "",
                ]
            )
            for img_entry in synthesis_images:
                img_path = self.output_dir / img_entry.file_path
                if img_path.exists():
                    try:
                        img = Image.open(img_path)
                        img_base64 = _image_to_base64(img)
                        lines.extend(
                            [
                                f"#### {img_entry.image_type}",
                                "",
                                f"![{img_entry.image_type}]({img_base64})",
                                "",
                            ]
                        )
                    except Exception as e:
                        lines.append(f"*Error loading image: {e}*")

        return "\n".join(lines)

    def _generate_images_section(self, image_entries: List[ImageLogEntry]) -> str:
        """Generate comprehensive images section."""
        lines = [
            "## All Images",
            "",
        ]

        # Group by stage
        by_stage: Dict[str, List[ImageLogEntry]] = {}
        for entry in image_entries:
            stage = entry.stage
            if stage not in by_stage:
                by_stage[stage] = []
            by_stage[stage].append(entry)

        for stage, entries in sorted(by_stage.items()):
            lines.extend(
                [
                    f"### {stage.capitalize()} Stage Images",
                    "",
                ]
            )

            for entry in entries:
                img_path = self.output_dir / entry.file_path
                if img_path.exists():
                    try:
                        img = Image.open(img_path)
                        img_base64 = _image_to_base64(img)
                        lines.extend(
                            [
                                f"#### {entry.image_type} - Step {entry.step_index or 'N/A'} - Bbox {entry.bbox_index or 'N/A'}",
                                "",
                                f"![{entry.image_type}]({img_base64})",
                                "",
                                f"- **Size:** {entry.original_size or 'N/A'}",
                            ]
                        )
                        if entry.cropped_size:
                            lines.append(f"- **Cropped Size:** {entry.cropped_size}")
                        if entry.crop_ratio:
                            lines.append(f"- **Crop Ratio:** {entry.crop_ratio:.2%}")
                        if entry.bbox:
                            lines.append(f"- **BBox:** {entry.bbox}")
                        lines.append(f"- **Timestamp:** {entry.timestamp}")
                        lines.append("")
                    except Exception as e:
                        lines.append(f"*Error loading image: {e}*")

        return "\n".join(lines)

    def _generate_timing_section(self, stage_traces: List[Any]) -> str:
        """Generate timing information section."""
        if not stage_traces:
            return ""

        lines = [
            "## Timing Information",
            "",
            "| Stage | Duration (ms) |",
            "|-------|---------------|",
        ]

        for stage in stage_traces:
            if hasattr(stage, "duration_ms"):
                lines.append(f"| {stage.stage_name} | {stage.duration_ms:.2f} |")

        lines.append("")
        return "\n".join(lines)

    def _generate_footer(self) -> str:
        """Generate Markdown footer."""
        return "\n---\n\n*Generated by CoRGI Pipeline Tracing System*"


__all__ = ["MarkdownReportGenerator"]
