# Giải thích các vấn đề trong Pipeline Output

## 1. Vấn đề Coordinates bị lặp lại

### Hiện tượng:
Tất cả 3 reasoning steps đều có cùng bounding box coordinates:
- Step 1: `[0.608, 0.083, 0.701, 0.886]` → `[1166, 89, 1345, 956]` pixels
- Step 2: `[0.608, 0.083, 0.701, 0.886]` → `[1166, 89, 1345, 956]` pixels  
- Step 3: `[0.608, 0.083, 0.701, 0.886]` → `[1166, 89, 1345, 956]` pixels

### Nguyên nhân:
1. **Batch Grounding Model Response**: Khi sử dụng batch grounding (single inference cho tất cả steps), model Qwen3-VL có thể:
   - Trả về cùng một region cho tất cả statements nếu chúng quá giống nhau
   - Không phân biệt được sự khác biệt giữa các statements
   - Statements quá generic: "Identify the 'Arena-Hard v2' category", "Compare the bar heights", "Verify the exact numerical value" - tất cả đều trỏ đến cùng một vùng trong biểu đồ

2. **Statement Quality**: Các statements từ reasoning step quá generic và không đủ cụ thể để model phân biệt:
   - "Identify the 'Arena-Hard v2' category on the x-axis" → quá rộng, có thể match toàn bộ chart
   - "Compare the bar heights for all models" → quá rộng
   - "Verify the exact numerical value" → quá rộng

### Giải pháp:
1. **Cải thiện Reasoning Prompts**: Yêu cầu model tạo ra các statements cụ thể hơn, chỉ định rõ vị trí (ví dụ: "the bar for OpenAI GPT-4o-0327 model in Arena-Hard v2 section")
2. **Per-step Grounding**: Thay vì batch grounding, có thể thử per-step grounding để model có thể focus vào từng statement riêng biệt
3. **Post-processing**: Thêm logic để detect và filter duplicate bboxes giữa các steps

## 2. Vấn đề Resolution ảnh cao

### Hiện tượng:
- Original image: `(1920, 1080)` pixels
- Cropped region: `(221, 875)` pixels từ bbox `[1166, 89, 1345, 956]`
- Evidence descriptions rất dài và chi tiết không cần thiết

### Nguyên nhân:
1. **Crop Logic**: Code đang crop đúng theo bbox, nhưng:
   - Bbox quá lớn (chiếm ~9% diện tích ảnh gốc)
   - Cropped image có aspect ratio dài (221x875 = rất hẹp và cao)
   - Không có resize sau khi crop trong một số trường hợp

2. **Model Response Length**: FastVLM và PaddleOCR đang generate responses quá dài:
   - FastVLM captioning: ~2000+ tokens cho một region nhỏ
   - PaddleOCR: Trả về markdown table format nhưng có thể có nhiều text không liên quan

3. **Resolution Limit**: Mặc dù có code để limit resolution (max 1024x1024), nhưng:
   - Code chỉ resize nếu `max(cropped_size) > max_dimension`
   - Với cropped size (221, 875), max dimension là 875 < 1024, nên không resize
   - Nhưng 875 pixels vẫn là khá cao cho một region nhỏ

### Giải pháp:
1. **Aggressive Resizing**: Resize cropped regions xuống nhỏ hơn (ví dụ: max 512x512) để:
   - Giảm token count
   - Tăng tốc inference
   - Giảm memory usage

2. **Response Length Limiting**: 
   - Giảm `max_new_tokens` cho captioning (hiện tại 700, có thể giảm xuống 200-300)
   - Thêm prompt instruction để model trả lời ngắn gọn hơn

3. **Better Crop Logic**: 
   - Thêm padding nhỏ xung quanh bbox để capture context
   - Nhưng vẫn resize về kích thước hợp lý

## 3. Vấn đề Evidence Descriptions không chính xác

### Hiện tượng:
Evidence descriptions rất dài và có nhiều thông tin không liên quan:
- Mô tả chi tiết về màu sắc, layout, background
- Nhiều suy đoán và phân tích không cần thiết
- Không focus vào thông tin cần thiết để trả lời câu hỏi

### Nguyên nhân:
1. **Prompt Design**: Captioning prompt không đủ cụ thể:
   - "Describe precisely what is visible" → quá mở, model sẽ mô tả mọi thứ
   - Thiếu instruction để focus vào thông tin liên quan đến statement

2. **Model Behavior**: FastVLM có xu hướng generate verbose descriptions

### Giải pháp:
1. **Improve Captioning Prompt**: 
   - Thêm instruction: "Focus only on information relevant to the statement: {statement}"
   - Yêu cầu: "Keep description under 2 sentences"
   - Thêm: "Do not describe colors, layout, or background unless directly relevant"

2. **Post-processing**: 
   - Truncate descriptions nếu quá dài
   - Extract key information từ long descriptions

## 4. Console Output không rõ ràng

### Vấn đề:
- Nhiều log messages trùng lặp
- Khó theo dõi flow của pipeline
- Thiếu thông tin về timing và performance

### Giải pháp:
1. **Structured Logging**: Sử dụng structured logging với levels rõ ràng
2. **Progress Indicators**: Thêm progress bars hoặc step indicators
3. **Summary Logging**: Log summary ở cuối mỗi stage thay vì log từng chi tiết

