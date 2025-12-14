"""
Configuration system for CoRGI pipeline.

Provides dataclasses for pipeline and model configuration,
with YAML loading support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import yaml

from ..utils.device_utils import normalize_device

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    model_id: str
    model_type: str
    device: str = ""  # Device must be specified in config (e.g., "cuda:7"), no default
    torch_dtype: str = "auto"  # "auto", "float16", "bfloat16", "float32"
    enable_compile: bool = False
    enable_flash_attn: bool = False
    use_cache: Optional[bool] = (
        None  # None = use model default (True), False = disable KV cache
    )
    max_image_size: int = (
        1024  # Maximum dimension for input images (maintain aspect ratio)
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelConfig:
        """Create ModelConfig from dictionary."""
        device = data.get("device", "")
        if not device:
            raise ValueError("device must be specified in config (e.g., 'cuda:7')")
        return cls(
            model_id=data.get("model_id", ""),
            model_type=data.get("model_type", ""),
            device=device,
            torch_dtype=data.get("torch_dtype", "auto"),
            enable_compile=data.get("enable_compile", False),
            enable_flash_attn=data.get("enable_flash_attn", False),
            use_cache=data.get("use_cache", None),
            max_image_size=data.get("max_image_size", 1024),
        )

    def __post_init__(self) -> None:
        """Validate and normalize device after initialization."""
        if not self.device:
            raise ValueError("device must be specified in config (e.g., 'cuda:0')")
        self.device = normalize_device(self.device, logger_override=logger)


@dataclass
class ReasoningConfig:
    """Configuration for reasoning stage."""

    model: ModelConfig
    max_steps: int = 3
    max_new_tokens: int = 512
    extraction_method: str = "hybrid"  # "regex", "llm", "hybrid"
    do_sample: bool = False  # Greedy decoding for faster inference
    temperature: float = 0.0  # Not used when do_sample=False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReasoningConfig:
        """Create ReasoningConfig from dictionary."""
        return cls(
            model=ModelConfig.from_dict(data.get("model", {})),
            max_steps=data.get("max_steps", 3),
            max_new_tokens=data.get("max_new_tokens", 512),
            extraction_method=data.get("extraction_method", "hybrid"),
            do_sample=data.get("do_sample", False),
            temperature=data.get("temperature", 0.0),
        )


@dataclass
class GroundingConfig:
    """Configuration for grounding stage."""

    model: ModelConfig
    max_regions: int = 1
    max_new_tokens: int = 128
    nms_enabled: bool = True
    nms_iou_threshold: float = 0.5
    reuse_reasoning: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "GroundingConfig":
        # If reusing reasoning, model config is optional
        reuse = data.get("reuse_reasoning", False)

        if reuse:
            # No model config needed when reusing
            return cls(
                model=None,  # Will be set to reasoning model later
                reuse_reasoning=True,
                max_regions=data.get("max_regions", 1),
                max_new_tokens=data.get("max_new_tokens", 128),
                nms_enabled=data.get("nms_enabled", True),
                nms_iou_threshold=data.get("nms_iou_threshold", 0.5),
            )
        else:
            # Standard grounding model
            return cls(
                model=ModelConfig.from_dict(data.get("model", {})),
                reuse_reasoning=False,
                max_regions=data.get("max_regions", 1),
                max_new_tokens=data.get("max_new_tokens", 128),
                nms_enabled=data.get("nms_enabled", True),
                nms_iou_threshold=data.get("nms_iou_threshold", 0.5),
            )


@dataclass
class CaptioningConfig:
    """Configuration for captioning stage."""

    model: ModelConfig
    max_new_tokens: int = 128
    # Image processing config (for Vintern)
    image_size: int = 448  # Vintern input size
    max_num_patches: int = 6  # Vintern max number of patches
    ocr_task: str = "ocr"  # PaddleOCR task type: 'ocr', 'table', 'chart', 'formula'
    # Optional separate OCR model (if not specified, use captioning model for OCR too)
    ocr_model: Optional[ModelConfig] = None
    # For composite model: separate OCR and Caption sub-configs
    ocr: Optional["CaptioningConfig"] = None
    caption: Optional["CaptioningConfig"] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CaptioningConfig:
        """Create CaptioningConfig from dictionary."""
        ocr_model_data = data.get("ocr_model")
        ocr_model = ModelConfig.from_dict(ocr_model_data) if ocr_model_data else None
        
        # Parse composite sub-configs (for composite model type)
        ocr_data = data.get("ocr")
        ocr_config = cls.from_dict(ocr_data) if ocr_data else None
        
        caption_data = data.get("caption")
        caption_config = cls.from_dict(caption_data) if caption_data else None
        
        return cls(
            model=ModelConfig.from_dict(data.get("model", {})),
            max_new_tokens=data.get("max_new_tokens", 128),
            image_size=data.get("image_size", 448),
            max_num_patches=data.get("max_num_patches", 6),
            ocr_task=data.get("ocr_task", "ocr"),
            ocr_model=ocr_model,
            ocr=ocr_config,
            caption=caption_config,
        )


@dataclass
class SynthesisConfig:
    """Configuration for synthesis stage."""

    model: Optional[ModelConfig]
    max_new_tokens: int = 384
    include_explanation: bool = True
    reuse_reasoning: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "SynthesisConfig":
        # If reusing reasoning, model config is optional
        reuse = data.get("reuse_reasoning", False)

        if reuse:
            return cls(
                model=None,  # Will be set to reasoning model later
                reuse_reasoning=True,
                max_new_tokens=data.get("max_new_tokens", 384),
                include_explanation=data.get("include_explanation", True),
            )
        else:
            return cls(
                model=ModelConfig.from_dict(data.get("model", {})),
                reuse_reasoning=False,
                max_new_tokens=data.get("max_new_tokens", 384),
                include_explanation=data.get("include_explanation", True),
            )


@dataclass
class CoRGiConfig:
    """Complete CoRGI pipeline configuration."""

    reasoning: ReasoningConfig
    grounding: GroundingConfig
    captioning: CaptioningConfig
    synthesis: SynthesisConfig

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> CoRGiConfig:
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(
            reasoning=ReasoningConfig.from_dict(data.get("reasoning", {})),
            grounding=GroundingConfig.from_dict(data.get("grounding", {})),
            captioning=CaptioningConfig.from_dict(data.get("captioning", {})),
            synthesis=SynthesisConfig.from_dict(data.get("synthesis", {})),
        )

    @classmethod
    def get_default(cls) -> CoRGiConfig:
        """Get default configuration."""
        # Note: Device must be specified in config file, not hardcoded here
        # This default is for testing only - production should use config files
        default_device = "cuda:7"
        return cls(
            reasoning=ReasoningConfig(
                model=ModelConfig(
                    model_id="Qwen/Qwen3-VL-2B-Instruct",
                    model_type="qwen_instruct",
                    device=default_device,
                ),
                max_steps=3,
                max_new_tokens=512,
                extraction_method="hybrid",
            ),
            grounding=GroundingConfig(
                model=ModelConfig(
                    model_id="microsoft/Florence-2-large",
                    model_type="florence2",
                    device=default_device,
                ),
                max_regions=1,
                max_new_tokens=128,
            ),
            captioning=CaptioningConfig(
                model=ModelConfig(
                    model_id="microsoft/Florence-2-large",
                    model_type="florence2",
                    device=default_device,
                ),
                max_new_tokens=128,
            ),
            synthesis=SynthesisConfig(
                model=ModelConfig(
                    model_id="Qwen/Qwen3-VL-2B-Instruct",
                    model_type="qwen_instruct",
                    device=default_device,
                ),
                max_new_tokens=256,
            ),
        )

    def unique_model_signatures(self) -> List[Tuple[str, str]]:
        """Get unique (model_type, model_id) tuples for all models."""
        signatures = []

        # Add reasoning model
        if self.reasoning and self.reasoning.model:
            signatures.append(
                (self.reasoning.model.model_type, self.reasoning.model.model_id)
            )

        # Add grounding model if not reusing reasoning
        if (
            self.grounding
            and self.grounding.model
            and not self.grounding.reuse_reasoning
        ):
            signatures.append(
                (self.grounding.model.model_type, self.grounding.model.model_id)
            )

        # Add captioning model
        if self.captioning and self.captioning.model:
            signatures.append(
                (self.captioning.model.model_type, self.captioning.model.model_id)
            )

        # Add synthesis model if not reusing reasoning
        if (
            self.synthesis
            and self.synthesis.model
            and not self.synthesis.reuse_reasoning
        ):
            signatures.append(
                (self.synthesis.model.model_type, self.synthesis.model.model_id)
            )

        return list(set(signatures))

    def requires_parallel_loading(self) -> bool:
        """
        Determine whether this configuration needs parallel loading.

        Returns True whenever multiple distinct model signatures are present.
        """
        return len(self.unique_model_signatures()) > 1

    def ensure_vintern_constraints(self) -> None:
        """
        Enforce Qwen + Vintern deployment constraints.

        If the captioning stage uses Vintern, phases 1 (reasoning),
        2 (grounding), and 4 (synthesis) must all share the same Qwen model.
        """
        if self.captioning.model.model_type != "vintern":
            return
        qwen_types = {"qwen_instruct", "qwen_thinking"}
        qwen_id = self.reasoning.model.model_id
        for stage_name, stage in (
            ("reasoning", self.reasoning),
            ("grounding", self.grounding),
            ("synthesis", self.synthesis),
        ):
            stage_model = stage.model
            if stage_model.model_type not in qwen_types:
                raise ValueError(
                    f"{stage_name.capitalize()} must use a Qwen model when captioning uses Vintern."
                )
            if stage_model.model_id != qwen_id:
                raise ValueError(
                    "Reasoning, grounding, and synthesis must share the same Qwen model_id "
                    "when deploying with Vintern."
                )


def load_config(yaml_path: str | Path) -> CoRGiConfig:
    """
    Load CoRGI configuration from YAML file.
    
    Convenience function that wraps CoRGiConfig.from_yaml().
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Returns:
        CoRGiConfig instance
        
    Example:
        >>> config = load_config("configs/default_v2.yaml")
    """
    return CoRGiConfig.from_yaml(yaml_path)


__all__ = [
    "ModelConfig",
    "ReasoningConfig",
    "GroundingConfig",
    "CaptioningConfig",
    "SynthesisConfig",
    "CoRGiConfig",
    "load_config",
]
