# CoRGI Deployment Testing Guide

This document describes how to test the CoRGI pipeline before deploying to Hugging Face Spaces.

## Prerequisites

1. **Environment Setup**:
   ```bash
   conda activate pytorch
   ```

2. **Install Dependencies**:
   ```bash
   cd /home/dungvpt/workspace/corgi_implementation/corgi_custom
   pip install -r requirements.txt
   ```

## Testing Steps

### 1. Test Individual Components

Test each component with the real Qwen3-VL model:

```bash
PYTHONPATH=/home/dungvpt/workspace/corgi_implementation/corgi_custom \
    conda run -n pytorch python test_components_debug.py
```

Expected output: All 6 tests should pass (✓).

### 2. Test CLI

Test the command-line interface with an example image:

```bash
cd /home/dungvpt/workspace/corgi_implementation/corgi_custom

# Download demo image if needed
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg -O /tmp/demo.jpeg

# Run CLI
PYTHONPATH=/home/dungvpt/workspace/corgi_implementation/corgi_custom \
    conda run -n pytorch python -m corgi.cli \
        --image /tmp/demo.jpeg \
        --question "How many people are there in the image? Is there any one who is wearing a white watch?" \
        --max-steps 3 \
        --max-regions 3
```

Expected output: Structured reasoning steps, visual evidence, and final answer.

### 3. Test Demo Script

Run the dedicated demo script:

```bash
PYTHONPATH=/home/dungvpt/workspace/corgi_implementation/corgi_custom \
    conda run -n pytorch python examples/demo_qwen_corgi.py \
        --max-steps 3 \
        --max-regions 3
```

Expected output: Similar to CLI test, with formatted output.

### 4. Test Gradio App (Optional)

Test the Gradio interface locally:

```bash
PYTHONPATH=/home/dungvpt/workspace/corgi_implementation/corgi_custom \
    conda run -n pytorch python app.py
```

Then open http://localhost:7860 in your browser and test:
1. Upload an image
2. Enter a question
3. Adjust max_steps and max_regions sliders
4. Click "Run CoRGI"
5. Check all tabs (Chain of Thought, ROI Extraction, Evidence, Answer, Performance)

### 5. Run Integration Tests

Run the official integration tests:

```bash
cd /home/dungvpt/workspace/corgi_implementation/corgi_custom

CORGI_RUN_QWEN_INTEGRATION=1 \
    PYTHONPATH=/home/dungvpt/workspace/corgi_implementation/corgi_custom \
    conda run -n pytorch python -m pytest corgi_tests/test_integration_qwen.py -v
```

Expected output: Tests should pass.

## Deployment

Once all tests pass, deploy to Hugging Face Spaces:

```bash
cd /home/dungvpt/workspace/corgi_implementation/corgi_custom

# Set your Hugging Face username (optional, defaults to tuandunghcmut)
export HF_USERNAME=tuandunghcmut
export HF_SPACE_NAME=corgi-qwen3-vl-demo

# Run deployment
./deploy_to_space.sh
```

The script will:
1. Check authentication
2. Create or update the Space
3. Copy all necessary files
4. Commit and push changes
5. Provide the Space URL

## Troubleshooting

### Issue: Model not found

**Solution**: Ensure you have access to Qwen3-VL models. Accept the model license on Hugging Face.

### Issue: CUDA out of memory

**Solution**: Try the smaller model:
```bash
export CORGI_QWEN_MODEL=Qwen/Qwen3-VL-4B-Instruct
```

### Issue: Import errors

**Solution**: Ensure PYTHONPATH is set correctly:
```bash
export PYTHONPATH=/home/dungvpt/workspace/corgi_implementation/corgi_custom
```

### Issue: Parser truncating statements

**Note**: This is a known issue when the model outputs verbose "thinking" text instead of structured JSON. The final answer is usually still correct. Future improvements will better handle this case.

## Known Limitations

1. **Thinking-mode outputs**: Qwen3-VL-8B-Thinking sometimes outputs verbose thinking processes instead of clean JSON, which can lead to truncated statement parsing. The pipeline still works, but displayed steps may be incomplete.

2. **Performance**: Full pipeline on CPU is slow (~60-70 seconds per query). Use GPU or smaller model for faster inference.

3. **ROI accuracy**: ROI extraction works best with clear, unambiguous visual queries. Complex spatial reasoning may require multiple steps.

## Next Steps

After successful deployment:

1. Monitor Space logs for any runtime errors
2. Test the Space URL to ensure it's working
3. Share the Space link with users
4. Collect feedback for improvements

## Support

For issues or questions:
- Check PROGRESS_LOG.md for recent changes
- Review QWEN_INFERENCE_NOTES.md for model usage tips
- See PROJECT_PLAN.md for project structure

