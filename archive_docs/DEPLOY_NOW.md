# 🚀 Quick Deploy Guide - Chỉ cần làm theo 3 bước!

## Tổng quan

Tất cả code đã sẵn sàng! Bạn chỉ cần chạy script deployment.

---

## Bước 1: Xác thực với Hugging Face

```bash
# Mở terminal và chạy
huggingface-cli login
```

Nhập token của bạn khi được yêu cầu (lấy từ https://huggingface.co/settings/tokens)

---

## Bước 2: (Tùy chọn) Cấu hình Space

```bash
# Mặc định sẽ dùng:
# - Username: tuandunghcmut
# - Space name: corgi-qwen3-vl-demo

# Nếu muốn đổi, export các biến này:
export HF_USERNAME=ten_cua_ban
export HF_SPACE_NAME=ten_space_cua_ban
```

---

## Bước 3: Deploy!

```bash
# Di chuyển vào thư mục
cd /home/dungvpt/workspace/corgi_implementation/corgi_custom

# Chạy script deploy
./deploy_to_space.sh
```

Script sẽ:
1. ✓ Kiểm tra authentication
2. ✓ Tạo hoặc cập nhật Space
3. ✓ Copy tất cả files cần thiết
4. ✓ Commit và push lên HuggingFace
5. ✓ Hiển thị URL của Space

---

## Sau khi deploy

1. **Đợi build hoàn thành** (~10-15 phút lần đầu)
   - Mở URL Space được cung cấp
   - Click tab "Logs" để xem tiến trình build

2. **Test Space**
   - Upload một ảnh
   - Nhập câu hỏi (VD: "How many people are in the image?")
   - Click "Run CoRGI"
   - Kiểm tra tất cả các tabs

3. **Chia sẻ!**
   - Space URL: `https://huggingface.co/spaces/{username}/{space_name}`
   - Default: `https://huggingface.co/spaces/tuandunghcmut/corgi-qwen3-vl-demo`

---

## Test trước khi deploy (Khuyến nghị)

```bash
# Test component
cd /home/dungvpt/workspace/corgi_implementation/corgi_custom
PYTHONPATH=$(pwd) conda run -n pytorch python test_components_debug.py

# Test demo
PYTHONPATH=$(pwd) conda run -n pytorch python examples/demo_qwen_corgi.py
```

---

## Lỗi thường gặp

### "Not logged in to Hugging Face"
**Giải pháp**: Chạy `huggingface-cli login`

### "Failed to clone space"
**Giải pháp**: Space chưa tồn tại hoặc không có quyền truy cập. Script sẽ tự động tạo Space mới.

### Build fails trên Space
**Kiểm tra**: 
- Logs tab trên Space để xem lỗi chi tiết
- Có thể do model cần accept license trên HuggingFace

---

## Thông tin quan trọng

### Model đang dùng
- **Model**: `Qwen/Qwen3-VL-8B-Thinking`
- **Kích thước**: ~16GB
- **Hiệu năng**: ~60-70 giây/query trên CPU

### Files được deploy
```
app.py                 # Entry point
requirements.txt       # Dependencies  
corgi/                 # Main code
examples/              # Demo scripts
README.md             # Documentation
```

### Tài liệu tham khảo
- `SUMMARY_REPORT.md` - Báo cáo tổng quan dự án
- `DEPLOYMENT_CHECKLIST.md` - Chi tiết từng bước deploy
- `USAGE_GUIDE.md` - Hướng dẫn sử dụng đầy đủ
- `TEST_DEPLOYMENT.md` - Hướng dẫn test

---

## Tóm lại

```bash
# 1. Login
huggingface-cli login

# 2. Deploy  
cd /home/dungvpt/workspace/corgi_implementation/corgi_custom
./deploy_to_space.sh

# 3. Đợi build xong và test!
```

**Đơn giản vậy thôi!** 🎉

---

## Cần trợ giúp?

1. Đọc `SUMMARY_REPORT.md` để hiểu overview
2. Đọc `DEPLOYMENT_CHECKLIST.md` cho chi tiết
3. Check `TEST_DEPLOYMENT.md` nếu muốn test trước

**Chúc may mắn!** 🚀

