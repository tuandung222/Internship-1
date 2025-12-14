#!/bin/bash

# Deploy CoRGI to Hugging Face Spaces
# Based on example_deploy_push.sh template

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}=========================================================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}=========================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Configuration
USERNAME="${HF_USERNAME:-tuandunghcmut}"
SPACE_NAME="${HF_SPACE_NAME:-corgi-qwen3-vl-florence-2-demo}"
SPACE_URL="https://huggingface.co/spaces/${USERNAME}/${SPACE_NAME}"
TEMP_DIR="/tmp/corgi-space-deploy"

print_header "🚀 Deploy CoRGI to Hugging Face Spaces"
echo ""
print_info "Space: ${USERNAME}/${SPACE_NAME}"
print_info "URL: ${SPACE_URL}"
echo ""

# Check if logged in
print_info "Checking authentication..."
if ! huggingface-cli whoami &> /dev/null; then
    print_error "Not logged in to Hugging Face"
    print_info "Please run: huggingface-cli login"
    exit 1
fi

CURRENT_USER=$(huggingface-cli whoami | head -1)
print_success "Authenticated as: ${CURRENT_USER}"

# Verify correct user
if [ "$CURRENT_USER" != "$USERNAME" ]; then
    print_warning "Current user (${CURRENT_USER}) != expected user (${USERNAME})"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Clean up old temp directory
if [ -d "$TEMP_DIR" ]; then
    print_info "Cleaning up old temporary directory..."
    rm -rf "$TEMP_DIR"
fi

# Create Space if it doesn't exist
print_info "Ensuring Space exists..."
python3 - <<EOF
from huggingface_hub import HfApi

api = HfApi()
try:
    api.create_repo(
        repo_id="${USERNAME}/${SPACE_NAME}",
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True
    )
    print("Space created or already exists")
except Exception as e:
    print(f"Error creating space: {e}")
    exit(1)
EOF

# Clone the space
print_info "Cloning Space repository..."
git clone "https://huggingface.co/spaces/${USERNAME}/${SPACE_NAME}" "$TEMP_DIR" &> /dev/null || {
    print_error "Failed to clone space"
    exit 1
}
print_success "Space cloned"

# Navigate to project directory
cd "$(dirname "$0")/../.."
CORGI_DIR=$(pwd)
print_info "CoRGI directory: ${CORGI_DIR}"

# Copy files to temp directory
print_info "Copying files..."

# Main files - app.py is now the Florence+Qwen version
print_info "  → app.py"
cp "${CORGI_DIR}/app.py" "$TEMP_DIR/"

print_info "  → requirements.txt"
cp "${CORGI_DIR}/requirements.txt" "$TEMP_DIR/"

print_info "  → README.md"
cp "${CORGI_DIR}/README.md" "$TEMP_DIR/"

# Copy corgi module
print_info "  → corgi/ module"
rm -rf "$TEMP_DIR/corgi"
cp -r "${CORGI_DIR}/corgi" "$TEMP_DIR/"

# Copy configs directory (required for app.py to find florence_qwen.yaml)
print_info "  → configs/ directory"
rm -rf "$TEMP_DIR/configs"
cp -r "${CORGI_DIR}/configs" "$TEMP_DIR/"

# Copy examples
if [ -d "${CORGI_DIR}/examples" ]; then
    print_info "  → examples/"
    rm -rf "$TEMP_DIR/examples"
    cp -r "${CORGI_DIR}/examples" "$TEMP_DIR/"
fi

# Copy documentation
for doc in PROJECT_PLAN.md PROGRESS_LOG.md QWEN_INFERENCE_NOTES.md; do
    if [ -f "${CORGI_DIR}/${doc}" ]; then
        print_info "  → ${doc}"
        cp "${CORGI_DIR}/${doc}" "$TEMP_DIR/"
    fi
done

print_success "Files copied"

# Navigate to temp directory
cd "$TEMP_DIR"

# Check if there are changes
print_info "Checking for changes..."
if git diff --quiet && git diff --cached --quiet; then
    print_warning "No changes detected. Nothing to deploy."
    rm -rf "$TEMP_DIR"
    exit 0
fi

# Show changes summary
print_info "Changes detected:"
git status --short | head -20
echo ""

# Stage all changes
print_info "Staging changes..."
git add .

# Create commit
COMMIT_MSG="Deploy CoRGI demo - $(date '+%Y-%m-%d %H:%M:%S')

Features:
- Structured reasoning with CoRGI protocol
- ROI extraction using Qwen3-VL grounding
- Visual evidence synthesis
- Gradio UI with per-step visualization

Model: Qwen/Qwen3-VL-8B-Thinking"

print_info "Creating commit..."
git commit -m "$COMMIT_MSG" > /dev/null
print_success "Commit created"

# Push to HF
print_info "Pushing to Hugging Face Spaces..."
if git push origin main 2>&1 | tee /tmp/push_output.log; then
    print_success "Push successful!"
else
    print_error "Push failed. Check output above."
    cat /tmp/push_output.log
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Cleanup
cd "$CORGI_DIR"
rm -rf "$TEMP_DIR"

print_success "Deployment complete!"
echo ""
print_header "🎉 Success!"
echo ""
print_info "Your Space is being rebuilt and will be available at:"
echo ""
echo -e "    ${GREEN}${SPACE_URL}${NC}"
echo ""
print_info "Building may take 5-10 minutes (model download + build)..."
print_info "Check status: ${SPACE_URL}/logs"
echo ""

# Offer to open in browser
read -p "Open Space in browser? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v xdg-open &> /dev/null; then
        xdg-open "${SPACE_URL}"
    elif command -v open &> /dev/null; then
        open "${SPACE_URL}"
    else
        print_info "Please open: ${SPACE_URL}"
    fi
fi

print_success "Done!"

