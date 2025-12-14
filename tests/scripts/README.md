# Test Scripts - Integration Tests

Thư mục này chứa các script test integration và end-to-end tests cho CoRGI pipeline.

## 📚 Danh sách scripts

- **[test_real_pipeline.py](test_real_pipeline.py)** - Test pipeline đầy đủ với real images
- **[test_flash_attn3.py](test_flash_attn3.py)** - Test Flash Attention 3
- **[test_structured_answer.py](test_structured_answer.py)** - Test structured answer format
- **[test_error_handling.py](test_error_handling.py)** - Test error handling
- **[test_components_debug.py](test_components_debug.py)** - Test từng component riêng lẻ
- **[test_single_gpu.py](test_single_gpu.py)** - Test trên single GPU
- **[batch_test.py](batch_test.py)** - Batch testing với nhiều test cases

## 🎯 Cách sử dụng

### Test pipeline đầy đủ
```bash
python test_scripts/test_real_pipeline.py --config configs/test_qwen_only.yaml --save-viz
```

### Test Flash Attention
```bash
python test_scripts/test_flash_attn3.py
```

### Test structured answer
```bash
python test_scripts/test_structured_answer.py
```

### Test error handling
```bash
python test_scripts/test_error_handling.py
```

### Test components
```bash
python test_scripts/test_components_debug.py
```

### Test single GPU
```bash
python test_scripts/test_single_gpu.py
```

### Batch test
```bash
python test_scripts/batch_test.py
```

## 📋 Test Suites

Ngoài các script này, còn có:
- **`corgi_tests/`** - Unit tests chính thức (chạy với pytest)
- **`tests/`** - Integration tests khác

## ⚠️ Lưu ý

- Một số test cần GPU
- Kiểm tra config files trong `configs/` trước khi chạy
- Kết quả test được lưu trong `test_results/`
