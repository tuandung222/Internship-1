# Benchmark Scripts

Thư mục này chứa các script để benchmark và đo performance của CoRGI.

## 📚 Danh sách scripts

- **[benchmark_optimizations.py](benchmark_optimizations.py)** - Benchmark các optimizations (Flash Attention, Torch Compile, etc.)

## 🎯 Cách sử dụng

### Benchmark optimizations
```bash
python scripts/benchmark/benchmark_optimizations.py
```

Script này sẽ benchmark:
- Flash Attention 2
- Torch Compile
- Greedy Decoding
- Các optimization khác

## 📊 Output

Kết quả benchmark sẽ hiển thị:
- Thời gian inference
- Memory usage
- Speedup so với baseline
- So sánh giữa các optimization methods

## ⚠️ Lưu ý

- Cần GPU để chạy benchmark chính xác
- Benchmark có thể mất thời gian
- Kết quả có thể khác nhau tùy vào hardware

