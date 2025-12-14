"""
HTML Report Generator for comprehensive pipeline tracing.

Generates an interactive HTML report with inline images, collapsible sections,
timeline visualization, and complete trace data.
"""

from __future__ import annotations

import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from io import BytesIO

from PIL import Image

from ..core.trace_types import PipelineTrace, ImageLogEntry, OutputTraceEntry


class HTMLReportGenerator:
    """Generates comprehensive HTML reports from trace data."""

    def __init__(self, output_dir: Path):
        """
        Initialize HTML report generator.

        Args:
            output_dir: Output directory for the report
        """
        self.output_dir = Path(output_dir)

    def generate_report(
        self,
        trace_data: PipelineTrace,
        image_logger,
        output_tracer,
    ) -> Path:
        """
        Generate comprehensive HTML report.

        Args:
            trace_data: Complete pipeline trace data
            image_logger: ImageLogger instance with all image entries
            output_tracer: OutputTracer instance with all trace entries

        Returns:
            Path to generated HTML file
        """
        html_content = self._generate_html(
            trace_data,
            image_logger.get_all_entries() if image_logger else [],
            output_tracer.get_all_entries() if output_tracer else [],
            output_tracer.get_all_stages() if output_tracer else [],
        )

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"trace_report_{timestamp}.html"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return report_path

    def _generate_html(
        self,
        trace_data: PipelineTrace,
        image_entries: List[ImageLogEntry],
        output_entries: List[OutputTraceEntry],
        stage_traces: List[Any],
    ) -> str:
        """Generate complete HTML content."""
        html_parts = [
            self._generate_header(trace_data),
            self._generate_summary_section(trace_data),
            self._generate_timeline_section(stage_traces),
            self._generate_images_section(image_entries),
            self._generate_stages_section(output_entries, image_entries),
            self._generate_footer(),
        ]
        return "\n".join(html_parts)

    def _generate_header(self, trace_data: PipelineTrace) -> str:
        """Generate HTML header with styles."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CoRGI Pipeline Trace Report - {trace_data.pipeline_id}</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>CoRGI Pipeline Trace Report</h1>
            <div class="metadata">
                <p><strong>Pipeline ID:</strong> {trace_data.pipeline_id}</p>
                <p><strong>Question:</strong> {trace_data.question}</p>
                <p><strong>Start Time:</strong> {trace_data.start_timestamp}</p>
                <p><strong>End Time:</strong> {trace_data.end_timestamp or 'N/A'}</p>
                <p><strong>Total Duration:</strong> {trace_data.total_duration_ms:.2f} ms</p>
            </div>
        </header>
"""

    def _generate_summary_section(self, trace_data: PipelineTrace) -> str:
        """Generate summary section."""
        config_str = (
            json.dumps(trace_data.config, indent=2) if trace_data.config else "{}"
        )
        return f"""
        <section class="summary">
            <h2>Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Configuration</h3>
                    <pre><code>{config_str}</code></pre>
                </div>
                <div class="summary-card">
                    <h3>Stages</h3>
                    <ul>
                        <li>Reasoning: {len([s for s in trace_data.stages if s.stage_type == 'reasoning'])}</li>
                        <li>Grounding: {len([s for s in trace_data.stages if s.stage_type == 'grounding'])}</li>
                        <li>Captioning: {len([s for s in trace_data.stages if s.stage_type == 'captioning'])}</li>
                        <li>Synthesis: {len([s for s in trace_data.stages if s.stage_type == 'synthesis'])}</li>
                    </ul>
                </div>
            </div>
        </section>
"""

    def _generate_timeline_section(self, stage_traces: List[Any]) -> str:
        """Generate timeline visualization."""
        if not stage_traces:
            return ""

        timeline_items = []
        for stage in stage_traces:
            duration_str = f"{stage.duration_ms:.1f}ms" if stage.duration_ms else "N/A"
            timeline_items.append(
                f"""
                <div class="timeline-item">
                    <div class="timeline-marker"></div>
                    <div class="timeline-content">
                        <h4>{stage.stage_name}</h4>
                        <p>Duration: {duration_str}</p>
                        <p>Type: {stage.stage_type}</p>
                    </div>
                </div>
            """
            )

        return f"""
        <section class="timeline">
            <h2>Pipeline Timeline</h2>
            <div class="timeline-container">
                {''.join(timeline_items)}
            </div>
        </section>
"""

    def _generate_images_section(self, image_entries: List[ImageLogEntry]) -> str:
        """Generate images gallery section."""
        if not image_entries:
            return ""

        # Group images by stage
        by_stage: Dict[str, List[ImageLogEntry]] = {}
        for entry in image_entries:
            stage = entry.stage
            if stage not in by_stage:
                by_stage[stage] = []
            by_stage[stage].append(entry)

        sections = []
        for stage, entries in by_stage.items():
            images_html = []
            for entry in entries:
                img_path = self.output_dir / entry.file_path
                if img_path.exists():
                    # Embed image as base64
                    try:
                        img = Image.open(img_path)
                        buffered = BytesIO()
                        img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        images_html.append(
                            f"""
                            <div class="image-card">
                                <h4>{entry.image_type} - Step {entry.step_index or 'N/A'} - Bbox {entry.bbox_index or 'N/A'}</h4>
                                <img src="data:image/png;base64,{img_str}" alt="{entry.image_type}">
                                <div class="image-metadata">
                                    <p><strong>Size:</strong> {entry.original_size or 'N/A'}</p>
                                    {f'<p><strong>Cropped Size:</strong> {entry.cropped_size}</p>' if entry.cropped_size else ''}
                                    {f'<p><strong>Crop Ratio:</strong> {entry.crop_ratio:.2%}</p>' if entry.crop_ratio else ''}
                                    {f'<p><strong>BBox:</strong> {entry.bbox}</p>' if entry.bbox else ''}
                                    <p><strong>Timestamp:</strong> {entry.timestamp}</p>
                                </div>
                            </div>
                        """
                        )
                    except Exception as e:
                        images_html.append(f"<p>Error loading image: {e}</p>")

            sections.append(
                f"""
                <div class="stage-images">
                    <h3>{stage.capitalize()} Images</h3>
                    <div class="image-gallery">
                        {''.join(images_html)}
                    </div>
                </div>
            """
            )

        return f"""
        <section class="images">
            <h2>Image Logs</h2>
            {''.join(sections)}
        </section>
"""

    def _generate_stages_section(
        self,
        output_entries: List[OutputTraceEntry],
        image_entries: List[ImageLogEntry],
    ) -> str:
        """Generate detailed stages section with outputs."""
        if not output_entries:
            return ""

        # Group by stage
        by_stage: Dict[str, List[OutputTraceEntry]] = {}
        for entry in output_entries:
            stage = entry.stage
            if stage not in by_stage:
                by_stage[stage] = []
            by_stage[stage].append(entry)

        sections = []
        for stage, entries in by_stage.items():
            stage_sections = []
            for entry in entries:
                raw_output_escaped = (
                    json.dumps(entry.raw_output, indent=2)
                    if entry.raw_output
                    else "N/A"
                )
                parsed_output_escaped = (
                    json.dumps(entry.parsed_output, indent=2)
                    if entry.parsed_output
                    else "N/A"
                )

                stage_sections.append(
                    f"""
                    <div class="trace-entry">
                        <h4>{stage.capitalize()} - Step {entry.step_index or 'N/A'} - Bbox {entry.bbox_index or 'N/A'}</h4>
                        <div class="trace-details">
                            <div class="trace-section">
                                <h5>Model</h5>
                                <p>{entry.model_name}</p>
                            </div>
                            <div class="trace-section">
                                <h5>Input Prompt</h5>
                                <pre><code>{entry.input_prompt or 'N/A'}</code></pre>
                            </div>
                            <details class="trace-section">
                                <summary>Raw Output</summary>
                                <pre><code>{raw_output_escaped}</code></pre>
                            </details>
                            <details class="trace-section">
                                <summary>Parsed Output</summary>
                                <pre><code>{parsed_output_escaped}</code></pre>
                            </details>
                            {self._format_intermediate_states(entry.intermediate_states)}
                            <div class="trace-section">
                                <h5>Metadata</h5>
                                <pre><code>{json.dumps(entry.metadata, indent=2)}</code></pre>
                            </div>
                            <div class="trace-section">
                                <p><strong>Duration:</strong> {entry.duration_ms:.2f} ms</p>
                                <p><strong>Timestamp:</strong> {entry.timestamp}</p>
                            </div>
                        </div>
                    </div>
                """
                )

            sections.append(
                f"""
                <div class="stage-section">
                    <h3>{stage.capitalize()} Stage</h3>
                    {''.join(stage_sections)}
                </div>
            """
            )

        return f"""
        <section class="stages">
            <h2>Stage Traces</h2>
            {''.join(sections)}
        </section>
"""

    def _format_intermediate_states(self, states: List[Dict[str, Any]]) -> str:
        """Format intermediate states as HTML."""
        if not states:
            return ""

        states_html = []
        for state in states:
            states_html.append(
                f"""
                <div class="intermediate-state">
                    <h5>Intermediate: {state.get('stage', 'Unknown')}</h5>
                    <pre><code>{json.dumps(state, indent=2)}</code></pre>
                </div>
            """
            )

        return f"""
            <details class="trace-section">
                <summary>Intermediate States</summary>
                {''.join(states_html)}
            </details>
"""

    def _generate_footer(self) -> str:
        """Generate HTML footer."""
        return """
    </div>
    <footer>
        <p>Generated by CoRGI Pipeline Tracing System</p>
    </footer>
</body>
</html>
"""

    def _get_css_styles(self) -> str:
        """Get CSS styles for the report."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        header {
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        h1 {
            color: #2c3e50;
            margin-bottom: 15px;
        }
        
        .metadata {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }
        
        .metadata p {
            margin: 5px 0;
        }
        
        section {
            margin: 40px 0;
            padding: 20px;
            background: #fafafa;
            border-radius: 6px;
        }
        
        h2 {
            color: #34495e;
            margin-bottom: 20px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        
        h3 {
            color: #2c3e50;
            margin: 20px 0 15px 0;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .summary-card {
            background: white;
            padding: 20px;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .summary-card pre {
            background: #f8f8f8;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }
        
        .timeline-container {
            position: relative;
            padding-left: 30px;
        }
        
        .timeline-item {
            position: relative;
            padding: 15px 0 15px 40px;
            margin-bottom: 20px;
            background: white;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .timeline-marker {
            position: absolute;
            left: -8px;
            top: 20px;
            width: 16px;
            height: 16px;
            background: #3498db;
            border-radius: 50%;
            border: 3px solid white;
            box-shadow: 0 0 0 3px #3498db;
        }
        
        .image-gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }
        
        .image-card {
            background: white;
            padding: 15px;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .image-card img {
            width: 100%;
            height: auto;
            border-radius: 4px;
            margin: 10px 0;
        }
        
        .image-metadata {
            font-size: 0.9em;
            color: #666;
        }
        
        .trace-entry {
            background: white;
            padding: 20px;
            margin: 15px 0;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .trace-section {
            margin: 15px 0;
            padding: 15px;
            background: #f8f8f8;
            border-radius: 4px;
        }
        
        .trace-section summary {
            cursor: pointer;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .trace-section pre {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 0.9em;
        }
        
        .trace-section code {
            font-family: 'Courier New', monospace;
        }
        
        .intermediate-state {
            margin: 10px 0;
            padding: 10px;
            background: #e8f4f8;
            border-radius: 4px;
        }
        
        footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            color: #666;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }
            
            .image-gallery {
                grid-template-columns: 1fr;
            }
        }
        """


__all__ = ["HTMLReportGenerator"]
