"""
Trace data structures for comprehensive pipeline logging and tracing.

Defines dataclasses to capture all information about pipeline execution:
- Image logging entries
- Output traces (raw, parsed, intermediate)
- Stage traces
- Complete pipeline traces
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class ImageLogEntry:
    """Metadata for a logged image."""
    
    stage: str  # 'reasoning', 'grounding', 'captioning', 'synthesis', 'original'
    step_index: Optional[int] = None
    bbox_index: Optional[int] = None
    image_type: str = "input"  # 'input', 'original', 'cropped', 'bbox_overlay'
    file_path: str = ""
    original_size: Optional[Tuple[int, int]] = None
    cropped_size: Optional[Tuple[int, int]] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    crop_ratio: Optional[float] = None  # Percentage of original image
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class OutputTraceEntry:
    """Trace entry for model output (raw, parsed, intermediate)."""
    
    stage: str  # 'reasoning', 'grounding', 'captioning', 'synthesis'
    step_index: Optional[int] = None
    bbox_index: Optional[int] = None
    model_name: str = ""
    model_config: Dict[str, Any] = field(default_factory=dict)
    input_prompt: Optional[str] = None
    input_parameters: Dict[str, Any] = field(default_factory=dict)
    raw_output: Optional[str] = None
    parsed_output: Optional[Any] = None
    intermediate_states: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Handle any non-serializable objects in parsed_output
        if result.get('parsed_output') is not None:
            try:
                import json
                json.dumps(result['parsed_output'])
            except (TypeError, ValueError):
                result['parsed_output'] = str(result['parsed_output'])
        return result


@dataclass
class StageTrace:
    """Complete trace for a pipeline stage."""
    
    stage_name: str
    stage_type: str  # 'reasoning', 'grounding', 'captioning', 'synthesis'
    step_index: Optional[int] = None
    start_timestamp: str = ""
    end_timestamp: str = ""
    duration_ms: float = 0.0
    image_entries: List[ImageLogEntry] = field(default_factory=list)
    output_entries: List[OutputTraceEntry] = field(default_factory=list)
    intermediate_states: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.start_timestamp:
            self.start_timestamp = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['image_entries'] = [entry.to_dict() for entry in self.image_entries]
        result['output_entries'] = [entry.to_dict() for entry in self.output_entries]
        return result


@dataclass
class PipelineTrace:
    """Complete trace for entire pipeline execution."""
    
    pipeline_id: str = ""
    question: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    start_timestamp: str = ""
    end_timestamp: str = ""
    total_duration_ms: float = 0.0
    stages: List[StageTrace] = field(default_factory=list)
    original_image_path: str = ""
    final_result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.pipeline_id:
            self.pipeline_id = f"pipeline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        if not self.start_timestamp:
            self.start_timestamp = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['stages'] = [stage.to_dict() for stage in self.stages]
        return result
    
    def save_json(self, file_path: Path) -> None:
        """Save trace to JSON file."""
        import json
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


__all__ = [
    "ImageLogEntry",
    "OutputTraceEntry",
    "StageTrace",
    "PipelineTrace",
]

