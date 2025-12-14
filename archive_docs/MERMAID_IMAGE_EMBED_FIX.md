# Fix: Mermaid Diagram Image Embedding on Hugging Face Spaces

## Problem

On Hugging Face Spaces, the Mermaid diagram image was not displaying in the introduction section. The issue was that:

1. **Gradio Markdown component limitations**: The `gr.Markdown` component doesn't reliably render external images or base64-encoded images embedded via HTML `<img>` tags.

2. **External URL issues**: Using external URLs (like `mermaid.ink`) may be blocked or fail to load in some environments.

## Solution

### 1. Convert Mermaid to Base64 Image

Changed `_mermaid_to_image_url()` to `_mermaid_to_image_base64()` which:
- Downloads the image from `mermaid.ink` API
- Converts it to base64-encoded data URI
- Returns a format like `data:image/png;base64,{base64_string}`

**File**: `corgi_custom/corgi/gradio_app.py`

```python
def _mermaid_to_image_base64(mermaid_code: str) -> Optional[str]:
    """
    Convert Mermaid diagram code to base64-encoded image using mermaid.ink API.
    """
    try:
        # Encode Mermaid code to base64 URL-safe
        mermaid_bytes = mermaid_code.encode('utf-8')
        mermaid_b64 = base64.urlsafe_b64encode(mermaid_bytes).decode('utf-8')
        
        # Create mermaid.ink URL
        mermaid_url = f"https://mermaid.ink/img/{mermaid_b64}"
        
        # Download image
        with urllib.request.urlopen(mermaid_url, timeout=10) as response:
            image_data = response.read()
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            return f"data:image/png;base64,{image_b64}"
    except Exception as e:
        logger.warning(f"Failed to convert Mermaid to image: {e}")
        return None
```

### 2. Rewrite Introduction in Pure HTML

**Problem**: Markdown rendering on Hugging Face Spaces was poor, especially:
- Tables with lists in cells (using `<br>` tags) didn't render correctly
- Code blocks had formatting issues
- Lists within paragraphs were broken

**Solution**: Completely rewrote `_generate_introduction_markdown()` as `_generate_introduction_html()` to generate pure HTML:
- Proper HTML tables with `<table>`, `<thead>`, `<tbody>`, `<tr>`, `<td>` tags
- Lists in table cells using proper `<ul><li>` structure
- Code blocks with proper `<pre><code>` tags and HTML escaping
- All content properly structured with semantic HTML

**Before** (Markdown):
```python
def _generate_introduction_markdown() -> str:
    sections.append("| Stage | Model | Purpose | Key Features |")
    sections.append("| **Stage 1: Reasoning** | Qwen3-VL-2B-Instruct | ... | - Chain-of-Thought reasoning<br>- JSON-structured output<br>- Noun phrase extraction |")
```

**After** (HTML):
```python
def _generate_introduction_html() -> str:
    html_parts.append('<table>')
    html_parts.append('<thead><tr><th>Stage</th><th>Model</th><th>Purpose</th><th>Key Features</th></tr></thead>')
    html_parts.append('<tbody>')
    html_parts.append('<tr>')
    html_parts.append('<td><strong>Stage 1: Reasoning</strong></td>')
    html_parts.append('<td>Qwen3-VL-2B-Instruct</td>')
    html_parts.append('<td>Generate structured reasoning steps</td>')
    html_parts.append('<td><ul><li>Chain-of-Thought reasoning</li><li>JSON-structured output</li><li>Noun phrase extraction</li></ul></td>')
    html_parts.append('</tr>')
    # ... more rows
    html_parts.append('</tbody>')
    html_parts.append('</table>')
```

### 3. Switch from Markdown to HTML Component

Changed the introduction section from `gr.Markdown` to `gr.HTML` to support:
- Base64-encoded images via `<img>` tags
- Full HTML rendering capabilities
- Proper table and list rendering

**Before**:
```python
introduction_markdown = gr.Markdown(
    value=_generate_introduction_markdown(),
    elem_classes=["corgi-introduction"],
)
```

**After**:
```python
introduction_html = _generate_introduction_html()
introduction_markdown = gr.HTML(
    value=_create_css_stylesheet() + f'<div class="corgi-introduction">{introduction_html}</div>',
)
```

### 4. HTML Escaping for Code Blocks

All code blocks now use proper HTML escaping:
```python
import html
html_parts.append('<pre><code class="language-text">')
html_parts.append(html.escape(example_prompt))
html_parts.append('</code></pre>')
```

This ensures special characters in prompts are properly displayed.

## Benefits

1. **Reliable Image Display**: Base64-encoded images work in all environments, including Hugging Face Spaces
2. **No External Dependencies**: Images are embedded directly, no need for external URL access
3. **Better Compatibility**: HTML component provides more control over rendering
4. **Graceful Fallback**: If image conversion fails, a note is displayed instead

## Testing

To verify the fix works:

1. **Local Testing**:
   ```bash
   cd corgi_custom
   python app.py
   ```
   Check that the Mermaid diagram image appears in the introduction section.

2. **Hugging Face Spaces**:
   - Deploy to Spaces
   - Verify the image displays correctly in the introduction section
   - Check browser console for any errors

## Files Modified

- `corgi_custom/corgi/gradio_app.py`:
  - **Rewrote** `_generate_introduction_markdown()` → `_generate_introduction_html()` to generate pure HTML
  - Changed `_mermaid_to_image_url()` to `_mermaid_to_image_base64()`
  - Changed introduction component from `gr.Markdown` to `gr.HTML`
  - Removed `_markdown_to_html()` function (no longer needed)

## Notes

- The Mermaid code block is still displayed for reference (highlighted with yellow border)
- The visual diagram appears directly below the code block
- CSS styling is preserved through the `corgi-introduction` class
- All HTML is properly escaped to prevent XSS vulnerabilities
- Tables now render correctly with proper list formatting in cells
- Code blocks use proper HTML escaping for special characters

