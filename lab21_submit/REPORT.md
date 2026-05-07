# Lab 21 - Báo Cáo Đánh Giá LoRA/QLoRA

**Học viên**: Vũ Minh Khải - 2A202600343  
**Ngày nộp**: 2026-05-07  
**Hình thức nộp**: Option A - Lightweight ZIP  

## 1. Setup

- **Model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **GPU**: Tesla T4, 15.6 GB VRAM
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, lấy 200 samples
- **Format**: Alpaca format, dùng các cột `instruction_vi`, `input_vi`, `output_vi`
- **Token length**: p95 = 562, chọn `max_seq_length = 1024`
- **Split**: 180 train / 20 eval
- **Training**: 3 epochs, batch size 1, gradient accumulation 8, effective batch size 8
- **LoRA**: target modules `q_proj`, `v_proj`
- **Tổng thời gian train**: 12.4 phút
- **Estimated cost**: $0.07

## 2. Kết Quả Rank Experiment

| Rank | Alpha | Trainable Params | Time | Peak VRAM | Eval Loss | Perplexity |
|------|-------|------------------|------|-----------|-----------|------------|
| 8 | 16 | 1,843,200 | 4.05 phút | 7.22 GB | 1.5577 | 4.7479 |
| 16 | 32 | 3,686,400 | 4.26 phút | 6.62 GB | 1.5161 | 4.5544 |
| 64 | 128 | 14,745,600 | 4.06 phút | 8.00 GB | 1.4768 | 4.3790 |

Perplexity được tính bằng `exp(eval_loss)`. Rank 64 có perplexity tốt nhất, nhưng số tham số trainable lớn hơn nhiều so với rank 16 và rank 8.

## 3. Loss Curve Analysis

Ảnh `loss_curve.png` là loss curve của baseline `r=16`. Train loss giảm từ khoảng 1.61 xuống khoảng 1.39, cho thấy model có học từ dataset.

Eval loss cuối của `r=16` là 1.5161, perplexity là 4.5544. Vì eval-during-training tắt trên T4, biểu đồ chủ yếu dùng train loss nên tôi chỉ kết luận ở mức cơ bản: chưa thấy dấu hiệu overfitting nghiêm trọng.

## 4. Qualitative Comparison

| # | Prompt | Nhận xét |
|---|--------|----------|
| 1 | Giải thích machine learning cho người mới bắt đầu | Fine-tuned trả lời rõ hơn, nhưng vẫn bị cắt ở cuối. |
| 2 | Viết code Python tính Fibonacci | Fine-tuned có cấu trúc code tốt hơn base, nhưng output chưa hoàn chỉnh. |
| 3 | Liệt kê 5 nguyên tắc UI/UX | Fine-tuned trả lời gọn hơn và dễ đọc hơn. |
| 4 | So sánh LoRA và QLoRA | Fine-tuned trả lời sai phần mở rộng LoRA, đây là case bị giảm chất lượng. |
| 5 | Phân biệt prompt engineering, RAG, fine-tuning | Fine-tuned mở đầu rõ hơn nhưng vẫn chưa đủ chi tiết. |

Kết quả qualitative cho thấy model fine-tuned thường trình bày có cấu trúc hơn. Tuy nhiên, model vẫn có lỗi factual ở câu LoRA/QLoRA, nên không thể chỉ dựa vào cảm giác câu trả lời trôi chảy.

## 5. Kết Luận Về Rank Trade-off

Rank 64 cho kết quả định lượng tốt nhất với perplexity 4.3790. Tuy nhiên, rank 64 có 14.7M trainable params, lớn hơn 4 lần so với rank 16. Mức cải thiện từ rank 16 lên rank 64 không quá lớn: perplexity giảm từ 4.5544 xuống 4.3790.

Nếu chỉ xét perplexity, tôi chọn rank 64. Nếu xét cân bằng giữa chất lượng, kích thước adapter và chi phí train, tôi chọn rank 16. Rank 8 nhẹ nhất nhưng chất lượng thấp nhất trong ba cấu hình.

## 6. What I Learned

- Rank cao hơn có thể cải thiện perplexity, nhưng adapter lớn hơn.
- Perplexity cần đi kèm qualitative evaluation vì model vẫn có thể trả lời sai kiến thức.
- Với GPU T4, QLoRA giúp fine-tune model 3B trong thời gian và VRAM chấp nhận được.
