# Scripts Directory

Thư mục này chứa tất cả các scripts hỗ trợ cho CoRGI project.

## 📁 Cấu trúc

```
scripts/
├── debug/              # Debug & inspect scripts
├── test/               # Test shell scripts
├── benchmark/          # Benchmark scripts
└── push_space.sh       # Deployment script (HuggingFace Spaces)
```

## 🎯 Các thư mục con

### 🔍 [debug/](debug/)
Scripts để debug và inspect các component:
- `debug_florence2.py`
- `deep_inspect_florence2.py`
- `inspect_florence_processor.py`

### 🧪 [test/](test/)
Shell scripts để chạy tests:
- `compare_qwen_florence.sh`
- `fix_florence2.sh`
- `test_florence2_quick.sh`

### 📊 [benchmark/](benchmark/)
Scripts để benchmark performance:
- `benchmark_optimizations.py`

### 🚀 Deployment
- `push_space.sh` - Push code lên HuggingFace Spaces

## 📝 Test Scripts

Các Python test scripts được tổ chức trong thư mục **`test_scripts/`** ở root:
- Integration tests
- End-to-end tests
- Component tests

Xem [test_scripts/README.md](../test_scripts/README.md) để biết thêm chi tiết.

## 🧪 Test Suites

Các test suite chính thức:
- **`corgi_tests/`** - Unit tests (chạy với `pytest`)
- **`tests/`** - Integration tests khác

## 💡 Tips

- Chạy từ root directory: `python scripts/debug/...`
- Hoặc từ scripts directory: `python debug/...`
- Shell scripts: `bash scripts/test/...` hoặc `chmod +x` và chạy trực tiếp

