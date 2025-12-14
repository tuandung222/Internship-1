"""CoRGI Models Package - Auto-register all model clients."""

# Import all model clients to trigger @ModelRegistry.register decorators
from . import florence  # noqa: F401
from . import qwen  # noqa: F401
from . import smolvlm  # noqa: F401
from . import fastvlm  # noqa: F401
from . import paddle  # noqa: F401
from . import vintern  # noqa: F401

# Export key classes
from .registry import ModelRegistry  # noqa: F401
from .factory import VLMClientFactory  # noqa: F401
from .composite.composite_captioning_client import CompositeCaptioningClient  # noqa: F401

__all__ = [
    "ModelRegistry",
    "VLMClientFactory",
    "CompositeCaptioningClient",
]

