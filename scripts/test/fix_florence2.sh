#!/bin/bash
# Fix Florence-2 compatibility issues

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Florence-2 Compatibility Fix Script                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate pytorch 2>/dev/null || true

echo "Current transformers version:"
pip show transformers | grep Version

echo ""
echo "Attempting Fix Method 1: Modify client code to use 'eager' attention"
echo "────────────────────────────────────────────────────────────────"
echo ""

# Backup original files
cp corgi/florence_grounding_client.py corgi/florence_grounding_client.py.bak 2>/dev/null || true
cp corgi/florence_captioning_client.py corgi/florence_captioning_client.py.bak 2>/dev/null || true

# Apply fix to grounding client
python << 'EOF'
import re

# Fix florence_grounding_client.py
with open('corgi/florence_grounding_client.py', 'r') as f:
    content = f.read()

# Find all from_pretrained calls and add attn_implementation="eager"
# Pattern 1: Without attn_implementation
pattern1 = r'(AutoModelForCausalLM\.from_pretrained\([^)]*?trust_remote_code=True,\s*torch_dtype=torch_dtype,)(\s*\))'
replacement1 = r'\1\n                attn_implementation="eager",\2'

# Pattern 2: With attn_implementation=None
pattern2 = r'attn_implementation=None,'
replacement2 = 'attn_implementation="eager",'

content = re.sub(pattern1, replacement1, content)
content = re.sub(pattern2, replacement2, content)

with open('corgi/florence_grounding_client.py', 'w') as f:
    f.write(content)

print("✅ Updated florence_grounding_client.py")

# Fix florence_captioning_client.py
with open('corgi/florence_captioning_client.py', 'r') as f:
    content = f.read()

content = re.sub(pattern1, replacement1, content)
content = re.sub(pattern2, replacement2, content)

with open('corgi/florence_captioning_client.py', 'w') as f:
    f.write(content)

print("✅ Updated florence_captioning_client.py")
EOF

echo ""
echo "Testing Florence-2 loading with eager attention..."

python -c "
from transformers import AutoModelForCausalLM
import torch

print('Loading Florence-2-large...')
model = AutoModelForCausalLM.from_pretrained(
    'microsoft/Florence-2-large',
    trust_remote_code=True,
    torch_dtype='auto',
    attn_implementation='eager'
)
print('✅ Florence-2 loaded successfully with eager attention!')
print(f'Model device: {next(model.parameters()).device}')
print(f'Model dtype: {next(model.parameters()).dtype}')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  ✅ Florence-2 Fix Applied Successfully!                       ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Changes made:"
    echo "  • Modified florence_grounding_client.py"
    echo "  • Modified florence_captioning_client.py"
    echo "  • Set attn_implementation='eager' for all Florence-2 loads"
    echo ""
    echo "Backup files created:"
    echo "  • corgi/florence_grounding_client.py.bak"
    echo "  • corgi/florence_captioning_client.py.bak"
    echo ""
    echo "Next step: Run the test!"
    echo "  ./test_florence2_quick.sh"
    echo ""
else
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  ⚠️  Fix Method 1 Failed                                       ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Trying Fix Method 2: Downgrade transformers..."
    echo "────────────────────────────────────────────────────────────────"
    echo ""
    
    # Restore backups
    cp corgi/florence_grounding_client.py.bak corgi/florence_grounding_client.py 2>/dev/null || true
    cp corgi/florence_captioning_client.py.bak corgi/florence_captioning_client.py 2>/dev/null || true
    
    pip install transformers==4.44.0
    
    echo ""
    echo "Testing Florence-2 with transformers 4.44.0..."
    
    python -c "
from transformers import AutoModelForCausalLM
print('Loading Florence-2-large with transformers 4.44.0...')
model = AutoModelForCausalLM.from_pretrained(
    'microsoft/Florence-2-large',
    trust_remote_code=True,
    torch_dtype='auto'
)
print('✅ Florence-2 loaded successfully!')
"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "╔════════════════════════════════════════════════════════════════╗"
        echo "║  ✅ Florence-2 Fixed with Transformers Downgrade!              ║"
        echo "╚════════════════════════════════════════════════════════════════╝"
        echo ""
        echo "Solution: Downgraded transformers to 4.44.0"
        echo ""
        echo "Next step: Run the test!"
        echo "  ./test_florence2_quick.sh"
        echo ""
    else
        echo ""
        echo "╔════════════════════════════════════════════════════════════════╗"
        echo "║  ❌ Both Fix Methods Failed                                    ║"
        echo "╚════════════════════════════════════════════════════════════════╝"
        echo ""
        echo "Please check FLORENCE2_TEST_PLAN.md for manual fixes."
        echo ""
        exit 1
    fi
fi

