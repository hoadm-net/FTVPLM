# Tinh chỉnh mô hình ngôn ngữ lớn tiếng Việt cho một số tác vụ xử lý ngôn ngữ tự nhiên.

## 1. Mục tiêu
* Tìm hiểu, so sánh 2 kỹ thuật tinh chỉnh phổ biến trong thời điểm hiện tại:
    * Tinh chỉnh toàn phần (full fine-tune - FFT)
    * LoRA (Low-Rank Adaptation of Large Language Models)
* Sử dụng nhiều GPUs để hỗ trợ tinh chỉnh các mô hình ngôn ngữ lớn đồng thời
* Tinh chỉnh mô hình ngôn ngữ lớn cho 2 tác vụ xử lý ngôn ngữ tự nhiên
    * Phân tích cảm xúc (Sentiment Analysis - SA)
    * Nhận dạng thực thể có tên (Named Entity Recognition - NER)

## 2. Datasets
* [UIT-VSFC](https://nlp.uit.edu.vn/datasets#h.p_4Brw8L-cbfTe)
* [PhoNER_COVID19](https://github.com/VinAIResearch/PhoNER_COVID19)

## 3. Các mô hình ngôn ngữ lớn cho tiếng Việt
* [PhoBERT-base-v2](https://huggingface.co/vinai/phobert-base-v2)
* [PhoBERT-large](https://huggingface.co/vinai/phobert-large)
* [BARTpho-word](https://huggingface.co/vinai/bartpho-word)
* [ViT5-base](https://huggingface.co/VietAI/vit5-base)

## 4. Fine-tuning Techniques
1. Full Model Fine-tuning
2. LoRA (Low-Rank Adaptation)

## 5. Cài đặt các gói phụ thuộc
* [Pytorch](https://pytorch.org/get-started/locally/)
* [Jupyter Notebook](https://jupyter.org/install) (Nếu cần)
* Cài đặt **Transformer** phiên bản mới nhất cho bài toán SA trên tập UIT-VSFC (để chạy ViT5 cho bài toán SequenceClassification)
* Cài đặt **Transformer** phiên bản có thể sử dụng Fast text cho PhoBERT theo hướng dẫn tại [Repo PhoBERT](https://github.com/VinAIResearch/PhoBERT) để chạy bài toán NER

```
pip install pandas
pip install datasets adapters peft bitsandbytes
pip install lightning torchmetrics
pip install underthesea
pip install matplotlib seaborn
pip install peft==0.5.0
```
