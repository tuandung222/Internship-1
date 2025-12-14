"""
CoRGI Core Module

Contains pipeline implementations, types, and streaming API.
"""

from .pipeline import CoRGIPipeline, PipelineResult
from .pipeline_v2 import CoRGIPipelineV2, PipelineResultV2
from .streaming import StreamEventType, StreamEvent, StreamingPipelineExecutor

__all__ = [
    # V1 Pipeline
    "CoRGIPipeline",
    "PipelineResult",
    # V2 Pipeline
    "CoRGIPipelineV2",
    "PipelineResultV2",
    # Streaming API
    "StreamEventType",
    "StreamEvent",
    "StreamingPipelineExecutor",
]
