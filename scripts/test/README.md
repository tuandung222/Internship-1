# Test Shell Scripts

Thư mục này chứa các shell script để chạy tests và so sánh models.

## 📚 Danh sách scripts

- **[compare_qwen_florence.sh](compare_qwen_florence.sh)** - So sánh Qwen và Florence-2
- **[fix_florence2.sh](fix_florence2.sh)** - Script fix các vấn đề với Florence-2
- **[test_florence2_quick.sh](test_florence2_quick.sh)** - Quick test cho Florence-2

## 🎯 Cách sử dụng

### So sánh Qwen và Florence-2
```bash
bash scripts/test/compare_qwen_florence.sh
```

### Fix Florence-2 issues
```bash
bash scripts/test/fix_florence2.sh
```

### Quick test Florence-2
```bash
bash scripts/test/test_florence2_quick.sh
```

## ⚠️ Lưu ý

- Đảm bảo có quyền thực thi: `chmod +x scripts/test/*.sh`
- Một số script có thể cần GPU
- Kiểm tra config files trước khi chạy

