# Lab02 — Sentiment Analysis với Deep Learning (aspect-level)

Bài giải Exercise trang 21 của slide Chapter 3. **Dùng đúng hai CSV aspect-level đã chuẩn hóa của bài trước, không chuyển thành phân loại cả câu.**

## Mở bài làm

- **`Lab02_5Fold_CV.ipynb`**: bản mới chạy đủ 5-fold CV, có kết quả từng fold và mean ± SD.
- **`BAO_CAO_CV.md`**: báo cáo đầy đủ 5-fold; dùng bản này để đọc kết quả CV.
- **`cross_validation/results/metrics_cv_summary.csv`**: mean và độ lệch chuẩn mẫu (`ddof=1`) của 10 phương pháp.
- **`Lab02_Aspect_Sentiment.ipynb`**: notebook tiếng Việt, có kết quả và biểu đồ đã chạy.
- **`BAO_CAO.md`**: báo cáo phương pháp, bảng số liệu thực tế, ablation và phân tích lỗi.
- **`results/metrics_test.csv`**: toàn bộ độ đo test của 10 mô hình/phương pháp.

Hai mục notebook/báo cáo cũ cùng thư mục `results/` là **thí nghiệm holdout trước**, được giữ để đối chiếu.
Kết quả 5-fold nằm riêng trong `cross_validation/`; không dùng nhầm điểm holdout làm điểm CV.

## Chạy đầy đủ 5-fold CV

```powershell
conda activate py310
cd Lab02
python run_cross_validation.py
```

Máy chưa có BERT cache: thêm `--allow-download`. Mặc định 20 epoch tối đa, patience 4, seed 42.
Script tự tiếp tục các fold hoàn thành nếu mã nguồn, cấu hình và dữ liệu khớp chữ ký lần chạy.
Nếu muốn một thí nghiệm với cấu hình khác, lưu trữ thư mục `cross_validation/` cũ trước khi chạy mới.
Cache vector tạm được dọn sau khi dùng; thêm `--keep-cache` nếu muốn giữ.

Mỗi outer fold có một **inner-validation riêng** để chọn epoch và trọng số ensemble; outer fold chỉ chấm điểm.
Mỗi mẫu trong 3.601 mẫu development được dự đoán outer đúng một lần. Các khía cạnh cùng câu luôn cùng nhóm.
Baseline và DL cùng học trên inner-fit (khoảng 64% development), không refit trên toàn outer-train sau early stopping.
Mean/std được tính từ 5 điểm outer, không phải lấy 5 lần điểm trên cùng test.

Sau khi chọn phương pháp bằng mean CV Macro-F1, lấy trung bình xác suất của 5 fold-model rồi chấm test 1.120 mẫu.
`metrics_same_test_by_fold.csv` chỉ để kiểm tra từng model trên **cùng test**, không phải cross-validation trên test.
Dự đoán bằng phương pháp được chọn qua CV:

```powershell
python predict_cv.py --text "The food was excellent but the service was slow." --aspect "service"
```

Thư mục `cross_validation/fold_01` đến `fold_05` lưu checkpoint, inner/outer splits, augmentation,
lịch sử học và trọng số ensemble riêng từng fold. Notebook CV mặc định hiển thị kết quả đã chạy.

## Đối chiếu yêu cầu

| Yêu cầu trong slide | Phần thực hiện |
| --- | --- |
| Build SA bằng DL | CNN, BiLSTM-Attention, CNN-BiLSTM-Attention |
| Accuracy, Precision, Recall, F1, ROC | Macro Precision/Recall/F1, weighted F1, ROC-AUC OvR, confusion matrices |
| So sánh baseline cùng dataset | Majority, Naive Bayes, Logistic Regression, calibrated Linear SVM; train lại cùng split |
| Data augmentation | Synonym replacement và restricted random deletion trên train; so sánh có/không augmentation |
| Feature engineering | TF-IDF và pretrained BERT token embeddings; so sánh với random embedding |
| Ensemble | Soft voting từ ba mạng DL độc lập; trọng số chọn bằng validation |

## Dữ liệu

`data/restaurant_aspect_train.csv` và `data/restaurant_aspect_test.csv` là bản sao **nguyên byte** từ thư mục `../data/` có sẵn, lần lượt 3.602 và 1.120 dòng.
Các cột đầu vào chính là `text`, `aspect_term`, `polarity`, `sentence_id`, `from`, `to`.
Không tải dataset khác, không tự tổng hợp nhãn cả câu, không dùng CSV sentence-level.
Chỉ đổi tên cột trong bộ nhớ cho dễ lập trình; không sửa hai file nguồn này.

Các khía cạnh cùng một câu được giữ chung partition để tránh rò rỉ, nhưng vẫn là các mẫu riêng.
Một mẫu train có câu trùng test bị loại khỏi phần huấn luyện; test giữ nguyên 1.120 mẫu.
Train/validation: 2.850/751 mẫu. Nhãn theo thứ tự cố định: `negative`, `neutral`, `positive`.
Các file `train.csv`, `validation.csv`, `test.csv`, `train_augmented.csv` là sản phẩm của pipeline, không thay thế nguồn chuẩn hóa.

## Môi trường và chạy lại

Các lệnh `run_experiment.py` dưới đây chạy **holdout cũ**. Để chạy đủ 5 fold, dùng `run_cross_validation.py` ở trên.

Máy hiện tại đã có môi trường Conda **py310** với PyTorch CUDA và BERT trong cache.
Chạy từ thư mục bài tập:

```powershell
conda activate py310
cd Lab02
python run_experiment.py
```

Hoặc không cần activate:

```powershell
& D:/miniconda3/envs/py310/python.exe Lab02/run_experiment.py
```

Trên máy mới, cài dependencies bằng `python -m pip install -r requirements.txt`, rồi chạy:

```powershell
python run_experiment.py --allow-download
```

`--allow-download` cho phép Transformers tải `bert-base-uncased` nếu chưa có cache (cần Internet).
Mặc định chỉ đọc model đã lưu cục bộ. Không cần API key hoặc dịch vụ trả phí.
PyTorch CPU vẫn chạy được (`--device cpu`), nhưng chậm hơn GPU; chọn bản PyTorch phù hợp driver nếu cần CUDA.
Mở notebook bằng Jupyter hoặc VS Code, chọn kernel `Python 3 (py310)`.
Notebook mặc định đọc kết quả có sẵn; đổi `RUN_TRAINING = True` để huấn luyện lại.

```powershell
python run_experiment.py --epochs 20 --patience 4 --seed 42
python -m unittest discover -s tests -v
```

Chạy lại sẽ ghi đè checkpoint/kết quả cùng tên trong `models/`, `results/` và cập nhật báo cáo.
`cache/` chứa vector BERT dạng float16, có thể chiếm hơn 1 GB và có thể xóa sau khi huấn luyện.
Nếu không có cache, pipeline tự tính lại. BERT gốc không được đóng gói trong Lab02; vẫn cần cache Hugging Face hoặc tải ở lần đầu trên máy mới.
`results/environment.json` lưu phiên bản đã thực chạy; thay phiên bản thư viện hoặc GPU có thể khiến điểm chênh nhẹ dù cùng seed.

## Dự đoán mẫu mới

```powershell
python predict.py --text "The food was excellent but the service was slow." --aspect "food"
python predict.py --text "The food was excellent but the service was slow." --aspect "service"
```

Đầu vào luôn cần **cả câu và aspect**. Mặc định dùng soft-voting ensemble đã lưu.
Có thể dùng `--model BERT_Hybrid_Augmented` hoặc tên một neural model khác.
Script dự đoán này phục vụ các mô hình DL; baseline đã lưu bằng joblib phục vụ tái sử dụng trong Python.

## Cấu trúc mã nguồn

| File/thư mục | Nội dung |
| --- | --- |
| `data_utils.py` | Đọc CSV, kiểm tra offset, chia theo nhóm, augmentation bảo vệ aspect/phủ định |
| `networks.py` | CNN, BiLSTM, masked attention, hybrid |
| `run_experiment.py` | Baseline, BERT features, training, early stopping, ensemble, metrics và đồ thị |
| `predict.py` | Nạp checkpoint và suy luận theo khía cạnh |
| `write_report.py` | Tạo báo cáo tiếng Việt từ số liệu đã đo |
| `build_notebook.py` | Tạo notebook trình bày bài làm |
| `tests/test_pipeline.py` | Kiểm tra rò rỉ, bảo toàn khía cạnh, padding, metric và dữ liệu nguồn |
| `models/` | Checkpoint tốt nhất trên validation, baseline joblib, trọng số ensemble |
| `results/` | Metrics, dự đoán, lỗi, lịch sử học, biểu đồ, cấu hình và log |

Kết quả tăng/giảm chất lượng đều được báo cáo đúng theo thí nghiệm; không giả định augmentation hoặc ensemble chắc chắn tốt hơn.
Nguồn lý thuyết và mô tả chi tiết nằm trong `BAO_CAO.md`.
