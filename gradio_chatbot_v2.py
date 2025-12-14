"""
CoRGI V2 Pipeline - Chatbot-Style Gradio Interface with Streaming.

This app displays pipeline execution step-by-step like a chatbot conversation,
streaming intermediate results (reasoning, bboxes, evidence) progressively.

Refactored to use the streaming API from corgi.core.streaming.
"""

import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import time
from pathlib import Path
from typing import Generator, List, Optional, Tuple

from corgi.core.pipeline_v2 import CoRGIPipelineV2
from corgi.core.streaming import StreamEventType, StreamEvent
from corgi.models.factory import VLMClientFactory
from corgi.core.config import load_config


# Global pipeline instance
pipeline: Optional[CoRGIPipelineV2] = None
current_config: Optional[str] = None


def load_pipeline(config_path: str) -> str:
    """Load pipeline with given config."""
    global pipeline, current_config

    try:
        config = load_config(config_path)
        client = VLMClientFactory.create_from_config(config, parallel_loading=True)
        pipeline = CoRGIPipelineV2(vlm_client=client)
        current_config = config_path

        return f"✅ Pipeline loaded!\nConfig: {Path(config_path).name}"
    except Exception as e:
        return f"❌ Failed to load pipeline: {str(e)}"


def draw_bbox_on_image(
    image: Image.Image,
    bbox: List[float],
    label: str = "",
    color: str = "red",
    width: int = 3,
) -> Image.Image:
    """Draw a single bounding box on image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Convert normalized bbox to pixel coordinates
    w, h = img.size
    x1, y1, x2, y2 = bbox
    x1_px = int(x1 * w)
    y1_px = int(y1 * h)
    x2_px = int(x2 * w)
    y2_px = int(y2 * h)

    # Draw rectangle
    draw.rectangle([x1_px, y1_px, x2_px, y2_px], outline=color, width=width)

    # Draw label if provided
    if label:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
            )
        except Exception:
            font = ImageFont.load_default()

        # Background for text
        text_bbox = draw.textbbox((x1_px, y1_px - 20), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1_px, y1_px - 20), label, fill="white", font=font)

    return img


def draw_multiple_bboxes(
    image: Image.Image,
    bboxes: List[Tuple[List[float], str, str]],
) -> Image.Image:
    """Draw multiple bounding boxes with labels.

    Args:
        image: Input image
        bboxes: List of (bbox, label, color) tuples
    """
    img = image.copy()
    for bbox, label, color in bboxes:
        img = draw_bbox_on_image(img, bbox, label, color)
    return img


def format_bbox(bbox: List[float]) -> str:
    """Format bbox for display."""
    return f"[{', '.join(f'{x:.2f}' for x in bbox)}]"


def stream_pipeline_execution(
    image: Image.Image,
    question: str,
    max_steps: int = 6,
    max_regions: int = 1,
) -> Generator[Tuple[List, Optional[Image.Image]], None, None]:
    """
    Stream pipeline execution step-by-step using the streaming API.

    Yields:
        Tuple of (chat_history, annotated_image)
    """
    if pipeline is None:
        yield [(None, "❌ Please load a pipeline first!")], None
        return

    chat_history = []
    original_image = image.copy()
    annotated_image = image.copy()
    all_bboxes = []
    key_evidence_bboxes = []

    # User message
    chat_history.append((f"📷 Image uploaded\n❓ Question: {question}", None))
    yield chat_history, original_image
    time.sleep(0.3)

    # Run pipeline with streaming
    step_count = 0
    evidence_count = 0

    for event in pipeline.run_streaming(image, question, max_steps, max_regions):
        event_type = event.type

        # === PIPELINE START ===
        if event_type == StreamEventType.PIPELINE_START:
            chat_history.append(
                (None, "🚀 **Starting CoRGI Pipeline V2**\n\n_Processing your question..._")
            )
            yield chat_history, annotated_image

        # === PHASE START ===
        elif event_type == StreamEventType.PHASE_START:
            phase = event.phase
            if phase == "reasoning_grounding":
                msg = "**Phase 1+2: Structured Reasoning + Grounding**\n\n"
                msg += "_Analyzing image and generating reasoning steps with bounding boxes..._"
            elif phase == "fallback_grounding":
                msg = f"**Fallback Grounding** (Step {event.step_index + 1})\n\n"
                msg += "_Finding bounding box for step that needs visual evidence..._"
            elif phase == "evidence_extraction":
                msg = "**Phase 3: Smart Evidence Routing**\n\n"
                msg += "_Extracting visual evidence from regions (OCR or Caption)..._"
            elif phase == "synthesis":
                msg = "**Phase 4: Answer Synthesis**\n\n"
                msg += "_Generating final answer from collected evidence..._"
            else:
                msg = f"**{phase.replace('_', ' ').title()}**"

            chat_history.append((None, msg))
            yield chat_history, annotated_image
            time.sleep(0.2)

        # === COT TEXT ===
        elif event_type == StreamEventType.COT_TEXT:
            cot_text = event.data.get("cot_text", "")
            cot_preview = cot_text[:400] + "..." if len(cot_text) > 400 else cot_text
            chat_history.append(
                (None, f"**Chain of Thought:**\n\n_{cot_preview}_")
            )
            yield chat_history, annotated_image
            time.sleep(0.3)

        # === STEP ===
        elif event_type == StreamEventType.STEP:
            step_count += 1
            data = event.data
            statement = data.get("statement", "")[:80]
            has_bbox = data.get("has_bbox", False)
            bbox = data.get("bbox")
            need_obj = data.get("need_object_captioning", False)
            need_ocr = data.get("need_text_ocr", False)

            evidence_type = "🔵 Object" if need_obj else "📝 Text" if need_ocr else "💭 Reasoning"
            bbox_status = "✅ Has bbox" if has_bbox else "⚠️ Need grounding"

            msg = f"**Step {step_count}:** {statement}...\n"
            msg += f"  Type: {evidence_type} | {bbox_status}"

            chat_history.append((None, msg))

            # Add bbox to visualization
            if has_bbox and bbox:
                color = "green" if need_obj else "blue" if need_ocr else "gray"
                all_bboxes.append((bbox, f"S{step_count}", color))
                annotated_image = draw_multiple_bboxes(original_image, all_bboxes)

            yield chat_history, annotated_image
            time.sleep(0.2)

        # === BBOX (fallback grounding) ===
        elif event_type == StreamEventType.BBOX:
            bbox = event.data.get("bbox")
            step_idx = event.step_index
            if bbox:
                msg = f"**Step {step_idx + 1}:** Found bbox {format_bbox(bbox)}"
                all_bboxes.append((bbox, f"S{step_idx + 1}", "yellow"))
                annotated_image = draw_multiple_bboxes(original_image, all_bboxes)
                chat_history.append((None, msg))
                yield chat_history, annotated_image

        # === EVIDENCE ===
        elif event_type == StreamEventType.EVIDENCE:
            evidence_count += 1
            data = event.data
            ev_type = data.get("evidence_type", "")
            bbox = data.get("bbox", [])

            if ev_type == "object":
                description = data.get("description", "")
                msg = f"**Evidence {evidence_count}** (Object 🔵)\n"
                msg += f"BBox: {format_bbox(bbox)}\n"
                msg += f"Description: _{description}_"
            else:  # text
                ocr_text = data.get("ocr_text", "")
                msg = f"**Evidence {evidence_count}** (Text 📝)\n"
                msg += f"BBox: {format_bbox(bbox)}\n"
                msg += f"OCR: _{ocr_text}_"

            chat_history.append((None, msg))
            yield chat_history, annotated_image
            time.sleep(0.2)

        # === PHASE END ===
        elif event_type == StreamEventType.PHASE_END:
            phase = event.phase
            duration = event.duration_ms

            if phase == "reasoning_grounding":
                data = event.data
                steps_count = data.get("steps_count", 0)
                with_bbox = data.get("steps_with_bbox", 0)
                msg = f"_Phase 1+2 complete: {steps_count} steps ({with_bbox} with bbox) in {duration/1000:.1f}s_"
            elif phase == "evidence_extraction":
                data = event.data
                ev_count = data.get("evidence_count", 0)
                obj_count = data.get("object_count", 0)
                text_count = data.get("text_count", 0)
                msg = f"_Phase 3 complete: {ev_count} evidence ({obj_count} object, {text_count} text) in {duration/1000:.1f}s_"
            elif phase == "synthesis":
                ke_count = event.data.get("key_evidence_count", 0)
                msg = f"_Phase 4 complete: {ke_count} key evidence in {duration/1000:.1f}s_"
            else:
                msg = f"_{phase.replace('_', ' ').title()} complete in {duration/1000:.1f}s_"

            chat_history.append((None, msg))
            yield chat_history, annotated_image
            time.sleep(0.2)

        # === ANSWER ===
        elif event_type == StreamEventType.ANSWER:
            answer = event.data.get("answer", "")
            explanation = event.data.get("explanation", "")

            msg = "## 🎯 Final Answer\n\n"
            msg += f"**{answer}**\n\n"

            if explanation:
                msg += f"**Explanation:**\n_{explanation}_"

            chat_history.append((None, msg))
            yield chat_history, annotated_image
            time.sleep(0.3)

        # === KEY EVIDENCE ===
        elif event_type == StreamEventType.KEY_EVIDENCE:
            data = event.data
            bbox = data.get("bbox", [])
            description = data.get("description", "")
            idx = event.step_index + 1 if event.step_index is not None else len(key_evidence_bboxes) + 1

            key_evidence_bboxes.append((bbox, f"Key{idx}", "orange"))

            msg = f"**Key Evidence {idx}:** {description}\n"
            msg += f"BBox: {format_bbox(bbox)}"

            chat_history.append((None, msg))

            # Update image with key evidence
            annotated_image = draw_multiple_bboxes(original_image, key_evidence_bboxes)
            yield chat_history, annotated_image

        # === PIPELINE END ===
        elif event_type == StreamEventType.PIPELINE_END:
            data = event.data
            total_ms = data.get("total_duration_ms", 0)
            steps = data.get("steps_count", 0)
            evidences = data.get("evidence_count", 0)

            msg = "---\n"
            msg += f"**✅ Pipeline Complete**\n"
            msg += f"- Steps: {steps}\n"
            msg += f"- Evidence items: {evidences}\n"
            msg += f"- Total time: {total_ms/1000:.2f}s"

            chat_history.append((None, msg))
            yield chat_history, annotated_image

        # === WARNING ===
        elif event_type == StreamEventType.WARNING:
            message = event.data.get("message", "Unknown warning")
            chat_history.append((None, f"⚠️ Warning: {message}"))
            yield chat_history, annotated_image

        # === ERROR ===
        elif event_type == StreamEventType.ERROR:
            message = event.data.get("message", "Unknown error")
            chat_history.append((None, f"❌ Error: {message}"))
            yield chat_history, annotated_image


def process_question(
    image: Image.Image,
    question: str,
    max_steps: int,
    max_regions: int,
    history: List,
) -> Generator:
    """Process question and stream results."""
    if image is None:
        yield history + [(None, "❌ Please upload an image first!")], None
        return

    if not question.strip():
        yield history + [(None, "❌ Please enter a question!")], None
        return

    # Stream pipeline execution
    for chat_hist, img in stream_pipeline_execution(image, question, max_steps, max_regions):
        yield chat_hist, img


# =============================================================================
# Gradio Interface
# =============================================================================

with gr.Blocks(title="CoRGI V2 - Streaming Pipeline") as demo:
    gr.Markdown("""
    # 🐕 CoRGI Pipeline V2 - Interactive Streaming Demo

    Watch the pipeline execute step-by-step in real-time! Each phase streams its progress like a chatbot conversation.

    **Pipeline Flow:**
    1. **Phase 1+2:** Structured Reasoning + Grounding (merged)
    2. **Phase 3:** Smart Evidence Routing (Object captioning OR OCR)
    3. **Phase 4:** Answer Synthesis with key evidence
    """)

    with gr.Row():
        with gr.Column(scale=1):
            # Configuration
            gr.Markdown("## ⚙️ Configuration")

            config_dropdown = gr.Dropdown(
                choices=[
                    "configs/qwen_only_v2.yaml",
                    "configs/qwen_florence2_smolvlm2_v2.yaml",
                ],
                value="configs/qwen_only_v2.yaml",
                label="Pipeline Config",
                info="Select V2 configuration",
            )

            load_btn = gr.Button("Load Pipeline", variant="primary")
            load_status = gr.Textbox(label="Status", lines=2, interactive=False)

            gr.Markdown("## 📥 Input")

            image_input = gr.Image(type="pil", label="Upload Image", height=300)

            question_input = gr.Textbox(
                label="Your Question",
                placeholder="What do you see in this image?",
                lines=2,
            )

            with gr.Row():
                max_steps = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=6,
                    step=1,
                    label="Max Reasoning Steps",
                )
                max_regions = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=1,
                    step=1,
                    label="Max Regions per Step",
                )

            submit_btn = gr.Button("🚀 Run Pipeline", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ Clear", variant="secondary")

        with gr.Column(scale=2):
            gr.Markdown("## 📊 Pipeline Execution")

            chatbot = gr.Chatbot(
                label="Streaming Pipeline Output",
                height=500,
                show_label=True,
            )

            gr.Markdown("## 🖼️ Annotated Image")

            output_image = gr.Image(
                label="Bounding Boxes & Key Evidence", type="pil", height=350
            )

    # Event handlers
    load_btn.click(fn=load_pipeline, inputs=[config_dropdown], outputs=[load_status])

    submit_btn.click(
        fn=process_question,
        inputs=[image_input, question_input, max_steps, max_regions, chatbot],
        outputs=[chatbot, output_image],
    )

    clear_btn.click(
        fn=lambda: ([], None, ""), outputs=[chatbot, output_image, question_input]
    )

    # Examples
    gr.Markdown("## 📝 Example Questions")
    gr.Examples(
        examples=[
            ["test_image.jpg", "What objects do you see in this image?"],
            ["test_image.jpg", "How many yellow taxis are visible?"],
            ["test_image.jpg", "Describe the weather and atmosphere in the scene."],
            ["test_image.jpg", "What architectural features are prominent?"],
        ],
        inputs=[image_input, question_input],
        label="Click to load example",
    )

    gr.Markdown("""
    ---
    ### ✨ Features
    - **Real-time streaming** of each pipeline phase
    - **Progressive bbox visualization** as they are generated
    - **Smart evidence routing** (automatic Object vs Text detection)
    - **Detailed explanations** with key evidence highlights
    - **Performance metrics** for each phase

    ### 🤖 Models
    - **Reasoning:** Qwen3-VL-2B/4B-Instruct (Phase 1+2 merged)
    - **Captioning:** SmolVLM2-500M or Qwen (configurable)
    - **OCR:** Florence-2-base-ft (optional)
    - **Synthesis:** Reuses reasoning model (memory efficient)
    """)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CoRGI V2 Streaming Chatbot Demo")
    parser.add_argument(
        "--config",
        default="configs/qwen_only_v2.yaml",
        help="Pipeline config (default: qwen_only_v2.yaml)",
    )
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860, help="Server port")

    args = parser.parse_args()

    # Pre-load pipeline
    print(f"Loading pipeline with config: {args.config}")
    status = load_pipeline(args.config)
    print(status)

    # Launch
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
