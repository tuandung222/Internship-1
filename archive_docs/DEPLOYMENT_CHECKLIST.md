# CoRGI Deployment Checklist

## Pre-Deployment Verification

- [x] Model configuration updated to `Qwen/Qwen3-VL-8B-Thinking`
- [x] Parser improvements implemented
- [x] Component tests passing
- [x] Demo script working with real model
- [x] Deployment script created and tested
- [ ] Gradio app tested locally (optional but recommended)
- [ ] Integration tests passing (optional but recommended)

## Deployment Steps

### 1. Authenticate with Hugging Face

```bash
# Login to Hugging Face CLI
huggingface-cli login

# Or set token as environment variable
export HF_TOKEN=your_token_here
```

### 2. Configure Space Settings (Optional)

```bash
# Set your Hugging Face username (default: tuandunghcmut)
export HF_USERNAME=your_username

# Set space name (default: corgi-qwen3-vl-demo)
export HF_SPACE_NAME=your_space_name
```

### 3. Run Deployment Script

```bash
cd /home/dungvpt/workspace/corgi_implementation/corgi_custom
./deploy_to_space.sh
```

The script will:
1. ✓ Check authentication
2. ✓ Create/update Space repository
3. ✓ Copy all necessary files
4. ✓ Commit changes
5. ✓ Push to Hugging Face
6. ✓ Provide Space URL

### 4. Monitor Build

After deployment:
1. Open the Space URL (provided by script)
2. Go to "Logs" tab
3. Watch the build process:
   - Dependencies installation (~2-3 minutes)
   - Model download (~5-10 minutes for first time)
   - App initialization

### 5. Test Deployed App

Once build completes:
1. Upload a test image
2. Enter a question
3. Adjust sliders (max_steps=3, max_regions=3)
4. Click "Run CoRGI"
5. Verify all tabs display correctly:
   - ✓ Chain of Thought
   - ✓ ROI Extraction
   - ✓ Evidence Descriptions
   - ✓ Answer Synthesis
   - ✓ Performance

## Space Configuration

### Hardware Requirements

**Recommended**: `cpu-basic` (free tier)
- Works but slower (~60-90 seconds per query)
- Model loads on-demand

**Alternative**: Request GPU upgrade
- Much faster (~10-20 seconds per query)
- Requires payment or community GPU grant

### Memory Requirements

- **4B Model**: ~8GB RAM
- **8B Model**: ~16GB RAM

If OOM errors occur:
1. Switch to 4B model in code
2. Or request upgraded hardware tier

## Post-Deployment

### Verify Functionality

- [ ] App loads without errors
- [ ] Image upload works
- [ ] Question input works
- [ ] Pipeline executes successfully
- [ ] Results display correctly
- [ ] All tabs show proper content

### Monitor Performance

Check Space logs for:
- Model loading time
- Inference time per query
- Memory usage
- Error rates

### Update README

Ensure Space README includes:
- [ ] Clear description
- [ ] Usage instructions
- [ ] Example queries
- [ ] Performance expectations
- [ ] Known limitations

## Files Deployed

The following files are copied to Space:

```
app.py                  # Main Gradio entrypoint
requirements.txt        # Python dependencies
README.md              # Space documentation
corgi/                 # Main package
  __init__.py
  cli.py
  gradio_app.py
  parsers.py
  pipeline.py
  qwen_client.py
  types.py
examples/              # Example scripts
  demo_qwen_corgi.py
PROJECT_PLAN.md        # Project overview
PROGRESS_LOG.md        # Development history
QWEN_INFERENCE_NOTES.md # Model usage tips
```

## Troubleshooting Deployment

### Issue: Authentication Failed

**Solution**:
```bash
huggingface-cli login
# Or provide token directly
export HF_TOKEN=your_token_here
```

### Issue: Space Already Exists

**Solution**: The script handles existing Spaces automatically. It will update the existing Space with new code.

### Issue: Git Push Failed

**Possible causes**:
1. Network issues - retry
2. Git credentials not configured
3. No write access to Space

**Solution**:
```bash
# Configure git if needed
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
```

### Issue: Build Fails on Space

**Check logs for**:
1. Missing dependencies → update requirements.txt
2. Import errors → check file structure
3. Model access denied → accept model license

### Issue: Runtime Errors

**Common fixes**:
1. CUDA errors → Switch to CPU-only
2. Memory errors → Use smaller model
3. Timeout errors → Increase gradio timeout

## Space Settings

Recommended settings in Space configuration:

```yaml
# Automatically configured by deployment script
sdk: gradio
sdk_version: "5.41.1"
python_version: "3.10"
app_file: app.py
```

## Updating Deployed Space

To update after making changes:

```bash
cd /home/dungvpt/workspace/corgi_implementation/corgi_custom

# Make your changes to code...

# Re-run deployment
./deploy_to_space.sh
```

The script will:
1. Detect changes
2. Commit with timestamp
3. Push update
4. Space will rebuild automatically

## Rollback

If deployment causes issues:

```bash
# Clone space locally
git clone https://huggingface.co/spaces/your_username/your_space_name
cd your_space_name

# Revert to previous commit
git log  # Find good commit hash
git revert <commit_hash>
git push origin main
```

## Success Criteria

Deployment is successful when:
- ✓ Space URL accessible
- ✓ Build completes without errors
- ✓ App loads and displays UI
- ✓ Can upload image and enter question
- ✓ Pipeline executes and returns results
- ✓ All tabs display proper content
- ✓ No runtime errors in logs

## Next Steps After Deployment

1. **Share**: Provide Space URL to users
2. **Monitor**: Check logs regularly for errors
3. **Iterate**: Collect feedback and improve
4. **Document**: Update README with examples
5. **Optimize**: Profile and improve performance

## Support

For issues during deployment:
- Check Space logs for detailed errors
- Review TEST_DEPLOYMENT.md for testing procedures
- See USAGE_GUIDE.md for configuration options
- Consult QWEN_INFERENCE_NOTES.md for model tips

## Maintenance

Regular maintenance tasks:
- [ ] Monitor Space usage/uptime
- [ ] Check for model updates
- [ ] Update dependencies periodically
- [ ] Review and respond to user feedback
- [ ] Document new features in README

---

**Ready to deploy?** Run `./deploy_to_space.sh` to begin! 🚀

