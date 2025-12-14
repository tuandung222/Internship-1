# CoRGI V2 - Streaming Chatbot Interface 🤖

An interactive Gradio chatbot that streams the CoRGI V2 pipeline execution step-by-step in real-time, showing intermediate results as they are generated.

## ✨ Features

### 🎥 Real-Time Streaming
- **Phase-by-phase execution** displayed like a chatbot conversation
- **Progressive visualization** of bounding boxes as they are generated
- **Live progress updates** for each reasoning step and evidence extraction
- **Timing information** for each phase

### 🎨 Interactive Visualization
- **Bounding boxes** drawn progressively on the image
- **Color-coded boxes**:
  - 🟢 **Green**: Object captioning regions
  - 🔵 **Blue**: OCR/text regions  
  - 🟠 **Orange**: Key evidence (final answer)
- **Labels** on each box showing step number

### 💬 Chatbot-Style Output
- Chain-of-thought reasoning preview
- Step-by-step reasoning with evidence type flags
- Progressive evidence extraction (one region at a time)
- Final answer with explanation and key evidence

### ⚙️ Configuration
- **Multiple configs**: Switch between Qwen-only or Multi-model setups
- **Adjustable parameters**: Max steps, max regions per step
- **Example questions**: Pre-loaded examples to try

---

## 🚀 Quick Start

### 1. Launch the App

```bash
# Basic launch
python gradio_chatbot_v2.py

# With specific config
python gradio_chatbot_v2.py --config configs/qwen_florence2_smolvlm2_v2.yaml

# With public sharing
python gradio_chatbot_v2.py --share

# Custom port
python gradio_chatbot_v2.py --server-port 8080
```

### 2. Use the Interface

1. **Load Pipeline**: Click "🔄 Load Pipeline" to initialize
2. **Upload Image**: Drag & drop or click to upload
3. **Ask Question**: Type your question in the text box
4. **Run Pipeline**: Click "🚀 Run Pipeline" to start
5. **Watch Streaming**: See each phase execute in real-time!

---

## 📊 Streaming Flow

### Example Conversation Flow

```
User: 🖼️ Image uploaded
      ❓ Question: What objects do you see in this image?

Bot:  🤔 Phase 1+2: Structured Reasoning + Grounding
      Analyzing image and generating reasoning steps with bounding boxes...

Bot:  💭 Chain of Thought:
      The question asks for a general description of objects in the image.
      I need to identify major visual elements: buildings, vehicles, people...

Bot:  📋 Generated 6 reasoning steps (6 with bounding boxes)
      
      Step 1: Identify the main setting as a busy urban street...
        ↳ Type: 🖼️ Object | ✅ Has bbox
      
      Step 2: Identify yellow taxis as key vehicles...
        ↳ Type: 🖼️ Object | ✅ Has bbox
      ...

Bot:  ⏱️ Phase 1+2 completed in 28.5s

Bot:  👁️ Phase 3: Smart Evidence Routing
      Extracting visual evidence from regions...

Bot:  🖼️ Step 1: Running object captioning...

Bot:  Region 1 (Object)
      📍 BBox: ['0.00', '0.00', '1.00', '0.80']
      📝 Description: A busy urban street with tall buildings and vehicles

Bot:  🖼️ Step 2: Running object captioning...

Bot:  Region 2 (Object)
      📍 BBox: ['0.36', '0.65', '0.72', '0.74']
      📝 Description: Yellow taxi cabs driving on the street

Bot:  ...

Bot:  ✅ Phase 3 complete: 6 evidence items extracted
        • Object evidence: 6
        • Text evidence: 0
      ⏱️ Completed in 2.3s

Bot:  ✍️ Phase 4: Answer Synthesis
      Generating final answer from evidence...

Bot:  🎉 Final Answer:
      
      A busy urban street in a modern city with tall skyscrapers,
      yellow taxis, pedestrians, and street infrastructure.
      
      💡 Explanation:
      The image captures a bustling cityscape with iconic yellow taxis,
      pedestrians crossing the street, and towering buildings...
      
      🔑 Key Evidence (3 regions):
      1. Yellow taxis driving down the street
         ↳ BBox: ['0.36', '0.65', '0.72', '0.74']
      
      2. Pedestrians walking on sidewalks
         ↳ BBox: ['0.00', '0.60', '0.30', '0.75']
      
      3. Tall skyscrapers and commercial signage
         ↳ BBox: ['0.85', '0.30', '1.00', '0.60']
      
      ⏱️ Phase 4 completed in 10.2s
      ⏱️ Total time: 41.0s
```

---

## 🎯 Use Cases

### 1. Document Understanding
```
Question: "What is the total amount on this invoice?"
→ Streams: Reasoning → OCR text regions → Extract numbers → Final answer
```

### 2. Scene Understanding
```
Question: "How many people are in the conference room?"
→ Streams: Reasoning → Object detection → Count people → Final answer
```

### 3. Multi-Step Reasoning
```
Question: "Which product has the highest price-to-performance ratio?"
→ Streams: Reasoning → Extract prices & specs → Calculate ratios → Compare
```

---

## ⚙️ Configuration Files

### Qwen-Only (Default)
```yaml
# configs/qwen_only_v2.yaml
- Reasoning: Qwen3-VL-4B-Instruct
- Grounding: Reuses reasoning model
- Captioning: Qwen captioning adapter
- Synthesis: Reuses reasoning model
```

**Pros**: 
- ✅ Single model (6GB VRAM)
- ✅ Consistent quality
- ✅ Fast startup

**Cons**:
- ⚠️ Captioning may be placeholder text (needs `_chat()` fix)

### Multi-Model (SmolVLM2 + Florence-2)
```yaml
# configs/qwen_florence2_smolvlm2_v2.yaml
- Reasoning: Qwen3-VL-2B-Instruct
- Grounding: Reuses reasoning model
- OCR: Florence-2-base-ft
- Captioning: SmolVLM2-500M-Video-Instruct
- Synthesis: Reuses reasoning model
```

**Pros**:
- ✅ Specialized models for each task
- ✅ Better OCR quality
- ✅ Faster captioning (SmolVLM2)

**Cons**:
- ⚠️ More VRAM (10-12GB)
- ⚠️ Longer startup time

---

## 🎨 UI Components

### Left Panel
- **Configuration**: Pipeline config selector
- **Load Button**: Initialize pipeline
- **Status**: Loading status messages
- **Image Upload**: Drag & drop area
- **Question Input**: Text box for question
- **Parameters**: Max steps and regions sliders
- **Run/Clear Buttons**: Execute or reset

### Right Panel
- **Chatbot**: Streaming conversation output
- **Annotated Image**: Progressive bbox visualization
- **Examples**: Pre-loaded example questions

---

## 🔧 Advanced Usage

### Custom Streaming Logic

You can modify `stream_pipeline_execution()` to customize what is streamed:

```python
def stream_pipeline_execution(...):
    # Add custom streaming points
    chat_history.append((None, "🔍 Custom step: Doing something special..."))
    yield chat_history, annotated_image
    
    # Your custom logic here
    ...
```

### Custom Bbox Colors

Modify `draw_bbox_on_image()` to use different colors:

```python
# In stream_pipeline_execution()
if step.need_object_captioning:
    color = "green"  # Change to any color
elif step.need_text_ocr:
    color = "blue"
else:
    color = "purple"  # Pure reasoning steps
```

### Add More Metadata

Stream additional information:

```python
# Show confidence scores
evidence_msg += f"🎯 Confidence: {confidence:.2%}\n"

# Show processing time per step
evidence_msg += f"⏱️ Processed in {duration:.2f}s\n"

# Show model used
evidence_msg += f"🤖 Model: {model_name}\n"
```

---

## 🐛 Troubleshooting

### Issue: Captioning shows "Region at (x, y, w, h)"

**Problem**: `QwenInstructClient` doesn't have `_chat()` method.

**Solution**: The captioning adapter falls back to placeholder text. To fix:

1. Implement `_chat()` in `QwenInstructClient`
2. Or use SmolVLM2 config which has proper captioning

### Issue: Slow streaming

**Problem**: Each step waits for model inference.

**Solution**: 
- Use faster models (2B instead of 4B)
- Reduce `max_steps` and `max_regions`
- Enable `torch.compile` in config

### Issue: Out of memory

**Problem**: Multiple models loaded.

**Solution**:
- Use `qwen_only_v2.yaml` config (single model)
- Set `reuse_reasoning: true` in all phases
- Reduce image resolution

---

## 📈 Performance

### Expected Timings (Qwen3-VL-4B, A100)

| Phase | Duration | % of Total |
|-------|----------|------------|
| **Phase 1+2** (Reasoning + Grounding) | 28-32s | 70% |
| **Phase 3** (Evidence extraction) | 0.1-3s | 5% |
| **Phase 4** (Synthesis) | 10-12s | 25% |
| **Total** | 40-45s | 100% |

### Optimization Tips

1. **Use Flash Attention 3**: 2-3x faster reasoning
2. **Enable Torch Compile**: Additional 1.5x speedup
3. **Reduce Image Resolution**: Set `max_pixels: 720`
4. **Fewer Steps**: Use `max_steps: 3-4` instead of 6
5. **Batch Processing**: Process multiple images together

---

## 🆚 Comparison: Chatbot vs Standard UI

| Feature | Chatbot UI | Standard UI |
|---------|-----------|-------------|
| **Streaming** | ✅ Real-time | ❌ All at once |
| **Progress Visibility** | ✅ Step-by-step | ⚠️ Loading spinner only |
| **User Engagement** | ✅ High | ⚠️ Medium |
| **Debugging** | ✅ Easy (see each step) | ❌ Hard |
| **Mobile Friendly** | ✅ Yes | ⚠️ Tables may overflow |
| **Implementation** | ⚙️ Custom streaming | 🔧 Standard Gradio |

---

## 🎓 Learning Resources

### Understanding the Flow
1. Read `docs/pipeline_v2/PIPELINE_V2_SUMMARY.md`
2. Check `docs/pipeline_v2/ARCHITECTURE_REVIEW_V2.md`
3. Review streaming code in `stream_pipeline_execution()`

### Customization Examples
- See `gradio_app.py` for standard UI
- Check `gradio_app_html.py` for HTML-based UI
- Compare implementations to understand trade-offs

---

## 🤝 Contributing

Want to improve the chatbot? Ideas:

1. **Add audio feedback**: TTS for each phase completion
2. **Add progress bars**: Visual progress within each phase
3. **Add export**: Download conversation as PDF/JSON
4. **Add comparison**: Side-by-side with different configs
5. **Add history**: Save previous conversations

---

## 📄 License

Apache 2.0 - Same as main CoRGI project

---

## 🙏 Acknowledgements

- **Gradio Team** for the amazing chatbot component
- **Qwen Team** for Qwen3-VL models
- **SmolVLM Team** for efficient VLM models
- **Florence Team** for grounding models

---

**Enjoy streaming your pipeline! 🚀**

For questions or issues, check the main README or open a GitHub issue.

