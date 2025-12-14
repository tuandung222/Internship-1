from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional, TextIO

from PIL import Image

from ..core.pipeline import CoRGIPipeline
from ..core.config import CoRGiConfig, ModelConfig
from ..models.factory import VLMClientFactory
from ..models.registry import ModelRegistry

# Legacy imports for backward compatibility
from ..models.qwen.qwen_client import Qwen3VLClient, QwenGenerationConfig
from ..models.florence.florence_client import Florence2Client

from ..core.types import GroundedEvidence, ReasoningStep

DEFAULT_MODEL_ID = (
    "Qwen/Qwen3-VL-2B-Instruct"  # Updated to 2B-Instruct for optimized performance
)
LEGACY_MODEL_ID = "Qwen/Qwen3-VL-4B-Thinking"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="corgi-cli",
        description="Run the CoRGI reasoning pipeline over an image/question pair.",
    )

    # Required arguments
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input image (jpg/png/etc.)",
    )
    parser.add_argument(
        "--question", type=str, required=True, help="Visual question for the image"
    )

    # Pipeline configuration
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3,
        help="Maximum number of reasoning steps to request",
    )
    parser.add_argument(
        "--max-regions",
        type=int,
        default=3,
        help="Maximum number of grounded regions per visual step",
    )

    # NEW: Config file support
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file (if provided, model args become overrides)",
    )

    # NEW: Model overrides
    parser.add_argument(
        "--reasoning-model",
        type=str,
        default=None,
        help="Override reasoning model (e.g., Qwen/Qwen3-VL-4B-Instruct)",
    )
    parser.add_argument(
        "--grounding-model",
        type=str,
        default=None,
        help="Override grounding model (e.g., microsoft/Florence-2-large)",
    )
    parser.add_argument(
        "--captioning-model",
        type=str,
        default=None,
        help="Override captioning model (uses grounding model if not specified)",
    )
    parser.add_argument(
        "--synthesis-model",
        type=str,
        default=None,
        help="Override synthesis model (uses reasoning model if not specified)",
    )

    # NEW: Extraction method
    parser.add_argument(
        "--extraction-method",
        choices=["regex", "llm", "hybrid"],
        default="hybrid",
        help="Method to extract structured steps from CoT (for Instruct models)",
    )

    # LEGACY: Backward compatibility arguments
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="[DEPRECATED] Legacy model ID (use --reasoning-model instead)",
    )
    parser.add_argument(
        "--use-florence",
        action="store_true",
        help="[SHORTCUT] Use Florence-2 for both grounding and captioning",
    )

    # Output options
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the pipeline result as JSON",
    )

    return parser


def _format_step(step: ReasoningStep) -> str:
    needs = "yes" if step.needs_vision else "no"
    suffix = f"; reason: {step.reason}" if step.reason else ""
    return f"[{step.index}] {step.statement} (needs vision: {needs}{suffix})"


def _format_evidence_item(evidence: GroundedEvidence) -> str:
    bbox = ", ".join(f"{coord:.2f}" for coord in evidence.bbox)
    parts = [f"Step {evidence.step_index} | bbox=({bbox})"]
    if evidence.description:
        parts.append(f"desc: {evidence.description}")
    if evidence.confidence is not None:
        parts.append(f"conf: {evidence.confidence:.2f}")
    return " | ".join(parts)


def _pipeline_factory_from_args(args) -> CoRGIPipeline:
    """
    Create pipeline from parsed CLI args.

    Supports both new config-based approach and legacy args.
    """
    # Load base config
    if args.config:
        print(f"Loading config from: {args.config}")
        config = CoRGiConfig.from_yaml(args.config)
    else:
        # Use defaults
        config = CoRGiConfig.get_default()

    # Apply CLI overrides
    if args.reasoning_model:
        config.reasoning.model.model_id = args.reasoning_model
        config.reasoning.model.model_type = ModelRegistry._detect_model_type(
            args.reasoning_model
        )

    # LEGACY: Handle --model-id
    if args.model_id:
        print("WARNING: --model-id is deprecated. Use --reasoning-model instead.")
        config.reasoning.model.model_id = args.model_id
        config.reasoning.model.model_type = ModelRegistry._detect_model_type(
            args.model_id
        )
        # For legacy, also use same model for synthesis
        config.synthesis.model.model_id = args.model_id
        config.synthesis.model.model_type = ModelRegistry._detect_model_type(
            args.model_id
        )

    # SHORTCUT: Handle --use-florence
    if args.use_florence:
        config.grounding.model.model_id = "microsoft/Florence-2-large"
        config.grounding.model.model_type = "florence2"
        config.captioning.model.model_id = "microsoft/Florence-2-large"
        config.captioning.model.model_type = "florence2"

    # Apply other model overrides
    if args.grounding_model:
        config.grounding.model.model_id = args.grounding_model
        config.grounding.model.model_type = ModelRegistry._detect_model_type(
            args.grounding_model
        )

    if args.captioning_model:
        config.captioning.model.model_id = args.captioning_model
        config.captioning.model.model_type = ModelRegistry._detect_model_type(
            args.captioning_model
        )
    elif args.grounding_model:
        # Use grounding model for captioning if not specified
        config.captioning.model.model_id = args.grounding_model
        config.captioning.model.model_type = ModelRegistry._detect_model_type(
            args.grounding_model
        )

    if args.synthesis_model:
        config.synthesis.model.model_id = args.synthesis_model
        config.synthesis.model.model_type = ModelRegistry._detect_model_type(
            args.synthesis_model
        )

    # Apply extraction method
    if args.extraction_method:
        config.reasoning.extraction_method = args.extraction_method

    # Apply max steps/regions from args
    config.reasoning.max_steps = args.max_steps
    config.grounding.max_regions = args.max_regions

    # Create client using factory
    print(f"Creating pipeline with:")
    print(
        f"  Reasoning: {config.reasoning.model.model_id} ({config.reasoning.model.model_type})"
    )
    print(
        f"  Grounding: {config.grounding.model.model_id} ({config.grounding.model.model_type})"
    )
    print(
        f"  Captioning: {config.captioning.model.model_id} ({config.captioning.model.model_type})"
    )
    print(
        f"  Synthesis: {config.synthesis.model.model_id} ({config.synthesis.model.model_type})"
    )

    client = VLMClientFactory.create_from_config(config)
    return CoRGIPipeline(vlm_client=client)


def _default_pipeline_factory(
    model_id: Optional[str], use_florence: bool = False, device: str = "cuda:7"
) -> CoRGIPipeline:
    """
    Legacy pipeline factory for backward compatibility.

    DEPRECATED: Use _pipeline_factory_from_args() instead.

    Args:
        model_id: Model ID (optional)
        use_florence: Whether to use Florence-2 (optional)
        device: Device to use (required, e.g., 'cuda:7')
    """
    if not device:
        raise ValueError("device must be specified (e.g., 'cuda:7')")

    config = QwenGenerationConfig(model_id=model_id or LEGACY_MODEL_ID, device=device)

    # OPTIMIZATION: Create Florence-2 client if requested
    florence_client = Florence2Client(device=device) if use_florence else None

    client = Qwen3VLClient(config=config, florence_client=florence_client)
    return CoRGIPipeline(vlm_client=client)


def execute_cli(
    *,
    image_path: Path,
    question: str,
    max_steps: int,
    max_regions: int,
    model_id: Optional[str],
    use_florence: bool = False,
    json_out: Optional[Path],
    pipeline_factory: Callable[[Optional[str], bool], CoRGIPipeline] | None = None,
    output_stream: TextIO | None = None,
) -> None:
    if output_stream is None:
        output_stream = sys.stdout
    factory = pipeline_factory or _default_pipeline_factory

    with Image.open(image_path) as img:
        image = img.convert("RGB")
        pipeline = factory(model_id, use_florence)
        result = pipeline.run(
            image=image,
            question=question,
            max_steps=max_steps,
            max_regions=max_regions,
        )

    print(f"Question: {question}", file=output_stream)
    print("-- Steps --", file=output_stream)
    for step in result.steps:
        print(_format_step(step), file=output_stream)
    if not result.steps:
        print("(no reasoning steps returned)", file=output_stream)

    print("-- Evidence --", file=output_stream)
    if result.evidence:
        for evidence in result.evidence:
            print(_format_evidence_item(evidence), file=output_stream)
    else:
        print("(no visual evidence)", file=output_stream)

    print("-- Answer --", file=output_stream)
    print(f"Answer: {result.answer}", file=output_stream)

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with json_out.open("w", encoding="utf-8") as handle:
            json.dump(result.to_json(), handle, ensure_ascii=False, indent=2)


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Use new factory to create pipeline from args
    pipeline = _pipeline_factory_from_args(args)

    execute_cli(
        image_path=args.image,
        question=args.question,
        max_steps=args.max_steps,
        max_regions=args.max_regions,
        pipeline_factory=lambda model_id, use_florence: pipeline,  # Pass pre-created pipeline
        json_out=args.json_out,
    )
    return 0


__all__ = ["build_parser", "execute_cli", "main"]

__all__ = ["build_parser", "execute_cli", "main"]
