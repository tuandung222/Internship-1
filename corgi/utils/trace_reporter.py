"""
CoRGI Trace Reporter

Generates comprehensive trace reports for pipeline execution including:
- Raw inputs/outputs for each component
- Intermediate visualizations
- Step-by-step reasoning explanation
- HTML report for debugging

Usage:
    from corgi.utils.trace_reporter import TraceReporter
    
    reporter = TraceReporter(output_dir="results/trace")
    
    # During pipeline execution, log each component
    reporter.log_phase_start("reasoning_grounding", {"question": question})
    reporter.log_phase_output("reasoning_grounding", {"cot": cot_text, "steps": steps})
    ...
    
    # Generate final report
    reporter.generate_html_report()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


@dataclass
class ComponentTrace:
    """Trace data for a single component."""
    
    component_name: str
    phase: str
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    
    # Raw input
    input_data: Dict[str, Any] = field(default_factory=dict)
    input_image_crop: Optional[str] = None  # Path to cropped input image
    
    # Raw output
    output_data: Dict[str, Any] = field(default_factory=dict)
    output_visualization: Optional[str] = None  # Path to visualization
    
    # Additional info
    model_id: Optional[str] = None
    prompt_used: Optional[str] = None
    raw_response: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_name": self.component_name,
            "phase": self.phase,
            "duration_ms": self.duration_ms,
            "model_id": self.model_id,
            "input_data": self.input_data,
            "input_image_crop": self.input_image_crop,
            "output_data": self.output_data,
            "output_visualization": self.output_visualization,
            "prompt_used": self.prompt_used,
            "raw_response": self.raw_response,
        }


@dataclass
class PipelineTrace:
    """Complete trace for a pipeline execution."""
    
    trace_id: str
    timestamp: str
    question: str
    image_path: str
    image_size: Tuple[int, int]
    pipeline_version: str
    config_path: Optional[str] = None
    
    # Components
    components: List[ComponentTrace] = field(default_factory=list)
    
    # Final result
    final_answer: Optional[str] = None
    final_explanation: Optional[str] = None
    
    # Timing
    total_duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "question": self.question,
            "image_path": self.image_path,
            "image_size": list(self.image_size),
            "pipeline_version": self.pipeline_version,
            "config_path": self.config_path,
            "components": [c.to_dict() for c in self.components],
            "final_answer": self.final_answer,
            "final_explanation": self.final_explanation,
            "total_duration_ms": self.total_duration_ms,
        }


class TraceReporter:
    """
    Comprehensive trace reporter for pipeline debugging.
    
    Captures:
    - Input/output for each component
    - Intermediate visualizations
    - Prompts and raw responses
    - Timing information
    """
    
    def __init__(
        self,
        output_dir: Path,
        save_crops: bool = True,
        save_visualizations: bool = True,
        save_prompts: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.save_crops = save_crops
        self.save_visualizations = save_visualizations
        self.save_prompts = save_prompts
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "crops").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "prompts").mkdir(exist_ok=True)
        
        # Current trace
        self.trace: Optional[PipelineTrace] = None
        self.current_component: Optional[ComponentTrace] = None
        self.original_image: Optional[Image.Image] = None
        
    def start_trace(
        self,
        question: str,
        image: Image.Image,
        image_path: str,
        pipeline_version: str,
        config_path: Optional[str] = None,
    ) -> None:
        """Start a new pipeline trace."""
        trace_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.trace = PipelineTrace(
            trace_id=trace_id,
            timestamp=datetime.now().isoformat(),
            question=question,
            image_path=image_path,
            image_size=image.size,
            pipeline_version=pipeline_version,
            config_path=config_path,
        )
        
        self.original_image = image.copy()
        
        # Save original image
        img_path = self.output_dir / "images" / "original.jpg"
        image.save(img_path, quality=95)
    
    def log_component_start(
        self,
        component_name: str,
        phase: str,
        input_data: Dict[str, Any],
        model_id: Optional[str] = None,
        prompt: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        """Log the start of a component execution."""
        self.current_component = ComponentTrace(
            component_name=component_name,
            phase=phase,
            start_time=time.monotonic(),
            input_data=input_data,
            model_id=model_id,
            prompt_used=prompt,
        )
        
        # Save input crop if bbox provided
        if bbox and self.original_image and self.save_crops:
            crop_path = self._save_crop(bbox, component_name)
            self.current_component.input_image_crop = str(crop_path)
        
        # Save prompt if provided
        if prompt and self.save_prompts:
            prompt_path = self.output_dir / "prompts" / f"{component_name}.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
    
    def log_component_end(
        self,
        output_data: Dict[str, Any],
        raw_response: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        """Log the end of a component execution."""
        if not self.current_component:
            return
        
        self.current_component.end_time = time.monotonic()
        self.current_component.duration_ms = (
            self.current_component.end_time - self.current_component.start_time
        ) * 1000
        self.current_component.output_data = output_data
        self.current_component.raw_response = raw_response
        
        # Save visualization if bbox provided
        if bbox and self.original_image and self.save_visualizations:
            viz_path = self._save_visualization(
                bbox,
                self.current_component.component_name,
                output_data.get("description") or output_data.get("ocr_text") or "",
            )
            self.current_component.output_visualization = str(viz_path)
        
        # Add to trace
        if self.trace:
            self.trace.components.append(self.current_component)
        
        self.current_component = None
    
    def log_final_result(
        self,
        answer: str,
        explanation: Optional[str] = None,
        total_duration_ms: float = 0.0,
    ) -> None:
        """Log the final pipeline result."""
        if self.trace:
            self.trace.final_answer = answer
            self.trace.final_explanation = explanation
            self.trace.total_duration_ms = total_duration_ms
    
    def _save_crop(
        self,
        bbox: Tuple[float, float, float, float],
        name: str,
    ) -> Path:
        """Save a cropped region of the image."""
        if not self.original_image:
            return Path()
        
        w, h = self.original_image.size
        x1, y1, x2, y2 = bbox
        
        # Convert normalized to pixel
        left = int(x1 * w)
        top = int(y1 * h)
        right = int(x2 * w)
        bottom = int(y2 * h)
        
        # Ensure valid bounds
        left = max(0, min(left, w - 1))
        top = max(0, min(top, h - 1))
        right = max(left + 1, min(right, w))
        bottom = max(top + 1, min(bottom, h))
        
        crop = self.original_image.crop((left, top, right, bottom))
        
        crop_path = self.output_dir / "crops" / f"{name}.jpg"
        crop.save(crop_path, quality=95)
        
        return crop_path
    
    def _save_visualization(
        self,
        bbox: Tuple[float, float, float, float],
        name: str,
        label: str = "",
    ) -> Path:
        """Save visualization with bbox drawn on image."""
        if not self.original_image:
            return Path()
        
        img = self.original_image.copy()
        draw = ImageDraw.Draw(img)
        
        w, h = img.size
        x1, y1, x2, y2 = bbox
        
        # Convert normalized to pixel
        left = int(x1 * w)
        top = int(y1 * h)
        right = int(x2 * w)
        bottom = int(y2 * h)
        
        # Draw rectangle
        draw.rectangle([left, top, right, bottom], outline="red", width=3)
        
        # Draw label
        if label:
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
                )
            except Exception:
                font = ImageFont.load_default()
            
            text = label[:50] + "..." if len(label) > 50 else label
            draw.rectangle([left, top - 20, left + len(text) * 8, top], fill="red")
            draw.text((left + 2, top - 18), text, fill="white", font=font)
        
        viz_path = self.output_dir / "visualizations" / f"{name}.jpg"
        img.save(viz_path, quality=95)
        
        return viz_path
    
    def save_trace_json(self) -> Path:
        """Save trace data as JSON."""
        if not self.trace:
            return Path()
        
        json_path = self.output_dir / "trace.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.trace.to_dict(), f, indent=2, ensure_ascii=False)
        
        return json_path
    
    def generate_html_report(self) -> Path:
        """Generate an HTML report for visual debugging."""
        if not self.trace:
            return Path()
        
        html_path = self.output_dir / "trace_report.html"
        
        # Build HTML
        html = self._build_html_report()
        
        html_path.write_text(html, encoding="utf-8")
        
        return html_path
    
    def _build_html_report(self) -> str:
        """Build the HTML report content."""
        if not self.trace:
            return ""
        
        components_html = ""
        for i, comp in enumerate(self.trace.components):
            components_html += self._build_component_html(comp, i + 1)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CoRGI Pipeline Trace - {self.trace.trace_id}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', sans-serif; background: #f5f5f5; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
        .header h1 {{ font-size: 24px; margin-bottom: 10px; }}
        .header .meta {{ display: flex; gap: 30px; font-size: 14px; opacity: 0.9; }}
        
        .section {{ background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .section h2 {{ color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px;
                       margin-bottom: 15px; }}
        
        .question {{ font-size: 18px; color: #333; background: #f8f9fa; padding: 15px;
                     border-left: 4px solid #667eea; border-radius: 4px; }}
        
        .component {{ border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 15px;
                      overflow: hidden; }}
        .component-header {{ background: #f8f9fa; padding: 15px; border-bottom: 1px solid #e0e0e0;
                             display: flex; justify-content: space-between; align-items: center; }}
        .component-header h3 {{ color: #333; font-size: 16px; }}
        .component-header .badge {{ background: #667eea; color: white; padding: 4px 12px;
                                    border-radius: 20px; font-size: 12px; }}
        .component-body {{ padding: 15px; }}
        
        .row {{ display: flex; gap: 20px; margin-bottom: 15px; }}
        .col {{ flex: 1; }}
        
        .data-block {{ background: #f8f9fa; border-radius: 6px; padding: 15px; }}
        .data-block h4 {{ color: #666; font-size: 12px; text-transform: uppercase;
                         margin-bottom: 10px; }}
        .data-block pre {{ font-size: 13px; white-space: pre-wrap; word-break: break-word;
                           background: #fff; padding: 10px; border-radius: 4px;
                           border: 1px solid #e0e0e0; max-height: 300px; overflow-y: auto; }}
        
        .image-preview {{ max-width: 100%; height: auto; border-radius: 6px;
                          border: 1px solid #e0e0e0; }}
        .image-container {{ text-align: center; }}
        .image-container img {{ max-height: 400px; }}
        
        .answer {{ font-size: 18px; color: #28a745; font-weight: bold;
                   background: #e8f5e9; padding: 20px; border-radius: 8px; }}
        .explanation {{ color: #666; margin-top: 15px; font-style: italic; }}
        
        .timing {{ display: flex; gap: 20px; }}
        .timing-item {{ background: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; }}
        .timing-item .value {{ font-size: 24px; font-weight: bold; color: #667eea; }}
        .timing-item .label {{ font-size: 12px; color: #666; margin-top: 5px; }}
        
        .collapsible {{ cursor: pointer; }}
        .collapsible:after {{ content: ' ▼'; font-size: 10px; }}
        .collapsible.active:after {{ content: ' ▲'; }}
        .content {{ display: none; }}
        .content.show {{ display: block; }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🐕 CoRGI Pipeline Trace Report</h1>
            <div class="meta">
                <span>📅 {self.trace.timestamp}</span>
                <span>🔧 Pipeline {self.trace.pipeline_version.upper()}</span>
                <span>⏱️ {self.trace.total_duration_ms/1000:.2f}s</span>
                <span>🖼️ {self.trace.image_size[0]}x{self.trace.image_size[1]}</span>
            </div>
        </div>
        
        <!-- Input -->
        <div class="section">
            <h2>📥 Input</h2>
            <div class="row">
                <div class="col">
                    <div class="image-container">
                        <img src="images/original.jpg" alt="Input Image" class="image-preview">
                    </div>
                </div>
                <div class="col">
                    <h4 style="margin-bottom: 10px; color: #666;">Question</h4>
                    <div class="question">{self.trace.question}</div>
                    
                    <h4 style="margin: 20px 0 10px 0; color: #666;">Config</h4>
                    <div class="data-block">
                        <pre>{self.trace.config_path or "Default config"}</pre>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Components -->
        <div class="section">
            <h2>🔧 Pipeline Components ({len(self.trace.components)} steps)</h2>
            {components_html}
        </div>
        
        <!-- Output -->
        <div class="section">
            <h2>📤 Final Output</h2>
            <div class="answer">
                {self.trace.final_answer or "No answer generated"}
            </div>
            {f'<div class="explanation">{self.trace.final_explanation}</div>' if self.trace.final_explanation else ''}
            
            <h4 style="margin: 20px 0 10px 0; color: #666;">Timing Breakdown</h4>
            <div class="timing">
                <div class="timing-item">
                    <div class="value">{self.trace.total_duration_ms/1000:.2f}s</div>
                    <div class="label">Total</div>
                </div>
                {self._build_timing_html()}
            </div>
        </div>
    </div>
    
    <script>
        document.querySelectorAll('.collapsible').forEach(item => {{
            item.addEventListener('click', function() {{
                this.classList.toggle('active');
                const content = this.nextElementSibling;
                content.classList.toggle('show');
            }});
        }});
    </script>
</body>
</html>
"""
        return html
    
    def _build_component_html(self, comp: ComponentTrace, index: int) -> str:
        """Build HTML for a single component."""
        input_json = json.dumps(comp.input_data, indent=2, ensure_ascii=False)
        output_json = json.dumps(comp.output_data, indent=2, ensure_ascii=False)
        
        crop_html = ""
        if comp.input_image_crop:
            crop_html = f'<img src="{comp.input_image_crop}" alt="Input Crop" class="image-preview" style="max-height: 200px;">'
        
        viz_html = ""
        if comp.output_visualization:
            viz_html = f'<img src="{comp.output_visualization}" alt="Visualization" class="image-preview" style="max-height: 200px;">'
        
        prompt_html = ""
        if comp.prompt_used:
            prompt_preview = comp.prompt_used[:500] + "..." if len(comp.prompt_used) > 500 else comp.prompt_used
            prompt_html = f'''
            <div class="data-block">
                <h4>Prompt Used</h4>
                <pre>{prompt_preview}</pre>
            </div>
            '''
        
        raw_response_html = ""
        if comp.raw_response:
            response_preview = comp.raw_response[:1000] + "..." if len(comp.raw_response) > 1000 else comp.raw_response
            raw_response_html = f'''
            <div class="data-block">
                <h4>Raw Model Response</h4>
                <pre>{response_preview}</pre>
            </div>
            '''
        
        return f'''
        <div class="component">
            <div class="component-header">
                <h3>Step {index}: {comp.component_name}</h3>
                <div>
                    <span class="badge">{comp.phase}</span>
                    <span style="margin-left: 10px; color: #666;">{comp.duration_ms:.0f}ms</span>
                </div>
            </div>
            <div class="component-body">
                <div class="row">
                    <div class="col">
                        <div class="data-block">
                            <h4>Input Data</h4>
                            <pre>{input_json}</pre>
                        </div>
                        {crop_html}
                    </div>
                    <div class="col">
                        <div class="data-block">
                            <h4>Output Data</h4>
                            <pre>{output_json}</pre>
                        </div>
                        {viz_html}
                    </div>
                </div>
                
                <h4 class="collapsible" style="margin-top: 15px; color: #666; cursor: pointer;">
                    📝 Detailed Logs (click to expand)
                </h4>
                <div class="content">
                    {prompt_html}
                    {raw_response_html}
                    <div class="data-block">
                        <h4>Model Info</h4>
                        <pre>{comp.model_id or "N/A"}</pre>
                    </div>
                </div>
            </div>
        </div>
        '''
    
    def _build_timing_html(self) -> str:
        """Build timing breakdown HTML."""
        if not self.trace:
            return ""
        
        # Group by phase
        phase_times: Dict[str, float] = {}
        for comp in self.trace.components:
            phase = comp.phase
            phase_times[phase] = phase_times.get(phase, 0) + comp.duration_ms
        
        html_parts = []
        for phase, duration in phase_times.items():
            html_parts.append(f'''
            <div class="timing-item">
                <div class="value">{duration/1000:.2f}s</div>
                <div class="label">{phase}</div>
            </div>
            ''')
        
        return "".join(html_parts)


__all__ = [
    "TraceReporter",
    "ComponentTrace",
    "PipelineTrace",
]
