"""Generate a Vietnamese report from measured outputs, never invented scores."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent


def markdown_table(frame, digits=4):
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"]*len(cols)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.{digits}f}" if isinstance(v, float) else str(v).replace("|", "/")
                                         for v in row) + " |")
    return "\n".join(lines)


def write_report():
    def read(name, folder="results"):
        return json.loads((ROOT / folder / name).read_text(encoding="utf-8"))
    test = pd.read_csv(ROOT / "results/metrics_test.csv")
    val = pd.read_csv(ROOT / "results/metrics_validation.csv")
    audit, cfg, env = read("data_audit.json"), read("config.json"), read("environment.json")
    ensemble = read("ensemble.json", "models")
    selected = read("selected_model.json", "models")["model"]
    scores = test.set_index("model")
    best = scores.loc[selected]
    baseline = val[val.model.str.startswith("TFIDF")].sort_values("f1_macro", ascending=False).iloc[0].model
    effects = []
    for label, before, after in [
        ("BERT embedding", "Random_Hybrid", "BERT_Hybrid"),
        ("Data augmentation", "BERT_Hybrid", "BERT_Hybrid_Augmented"),
        ("Soft voting so với hybrid tăng cường", "BERT_Hybrid_Augmented", "SoftVoting_Ensemble")]:
        effects.append({"Thí nghiệm": label, "Mốc so sánh": before, "Biến thể": after,
                        "Delta Accuracy (điểm %)": 100*(scores.loc[after, "accuracy"]-scores.loc[before, "accuracy"]),
                        "Delta Macro F1 (điểm %)": 100*(scores.loc[after, "f1_macro"]-scores.loc[before, "f1_macro"])})
    distribution = pd.DataFrame([{"Tập": k, "Mẫu khía cạnh": v["rows"], "Câu khác nhau": v["sentences"],
                                  **v["classes"]} for k, v in audit["splits"].items()])
    joined = val[["model", "f1_macro"]].rename(columns={"f1_macro": "validation_f1"}).merge(test, on="model")
    joined = joined.sort_values("validation_f1", ascending=False)
    per_class = read(f"classification_report_{selected}.json")
    per_class = pd.DataFrame([{ "Nhãn": label, **per_class[label]} for label in ["negative", "neutral", "positive"]])
    mistakes = pd.read_csv(ROOT / "results" / f"errors_{selected}.csv")
    error_counts = mistakes.groupby(["label", "predicted"]).size().sort_values(ascending=False)
    error_summary = "; ".join(f"{truth} → {pred}: {count}" for (truth, pred), count in error_counts.items())
    example_lines = []
    for row in mistakes.head(5).itertuples():
        example_lines.append(f'- Câu: “{row.text}” — khía cạnh **{row.aspect}**; nhãn thật `{row.label}`, dự đoán `{row.predicted}`.')
    augmentation_delta = effects[1]["Delta Macro F1 (điểm %)"]
    augmentation_comment = ("cải thiện" if augmentation_delta > 0 else "giảm" if augmentation_delta < 0 else "không thay đổi")
    text = f"""# Lab02 — Phân tích cảm xúc theo khía cạnh bằng Deep Learning

## 1. Bài toán và dữ liệu

Thực hiện Exercise trang 21 của slide **Chapter 3: Sentiment Analysis project implementation**.
Đầu vào là một cặp **(review, aspect_term)**; đầu ra là `negative`, `neutral` hoặc `positive`.
Ví dụ, “The food was excellent but the service was slow” có thể mang hai nhãn khác nhau khi aspect là `food` và `service`.
Đây là **aspect polarity classification**, giả định đã biết aspect term; không làm aspect extraction và không phân loại toàn câu.

Dùng nguyên hai CSV đã chuẩn hóa của bài trước, sao chép byte-for-byte vào `Lab02/data/`:
`restaurant_aspect_train.csv` ({audit['raw_train_aspects']} dòng) và
`restaurant_aspect_test.csv` ({audit['raw_test_aspects']} dòng).
Giữ nguyên nhãn `polarity`, câu `text`, khía cạnh `aspect_term` và vị trí `from/to`.
Các CSV đầu vào đã không chứa `conflict`; pipeline không tự suy ra hoặc gộp nhãn mới.
Mã SHA-256 của dữ liệu được lưu trong `results/data_audit.json` để đối chiếu nguồn.

Phát hiện {audit['removed_train_overlap']} mẫu train có câu trùng test sau khi chuẩn hóa khoảng trắng/chữ thường;
loại những mẫu đó khỏi phần train, **giữ nguyên toàn bộ test**. Cách chia validation:
fold đầu của `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state={cfg['seed']})`,
group là nội dung câu chuẩn hóa. Chỉ gom nhóm khi chia tập để tránh rò rỉ;
mỗi dòng khía cạnh và nhãn riêng vẫn được giữ lại. Đây là một holdout, không phải kết quả trung bình 5-fold CV.

{markdown_table(distribution)}

## 2. Tiền xử lý và đặc trưng

- TF-IDF unigram/bigram, `min_df=2`, tối đa 30.000 đặc trưng; fit trên train.
  Đầu vào gồm câu, aspect lặp hai lần và cửa sổ 5 từ hai phía aspect, tương tự cách tạo đặc trưng ở bài trước.
- WordPiece cho các mô hình DL, giữ phủ định và dấu câu. Không loại stopwords hoặc stemming trước BERT.
- BERT dùng cặp `[CLS] aspect [SEP] review [SEP]`, tối đa {cfg['max_length']} token;
  chỉ cắt phần review khi quá dài để luôn giữ aspect. Số mẫu bị cắt được ghi trong `bert_features.json`.
- `bert-base-uncased` được **đóng băng**, lấy vector ngữ cảnh từng token (768 chiều) ở tầng cuối;
  lớp chiếu học được đưa về 96 chiều. Đây là feature extraction, không fine-tune toàn bộ BERT.
- `Random_Hybrid` dùng cùng WordPiece và cùng mạng phía sau, embedding 96 chiều khởi tạo ngẫu nhiên.
  So sánh này có thêm khác biệt về lớp chiếu và số tham số, chưa phải ablation kiểm soát hoàn toàn mọi yếu tố.

## 3. Các mô hình

1. Majority (xác suất theo tần suất train), Multinomial Naive Bayes, Logistic Regression và Linear SVM.
   SVM có hiệu chỉnh xác suất bằng 3-fold theo nhóm **bên trong train**; TF-IDF cũng fit riêng trong từng fold.
   Các baseline được huấn luyện lại trên cùng split với DL, không chép điểm từ notebook cũ.
2. CNN: ba nhánh convolution kernel 3/5/7, mỗi nhánh 64 kênh, ReLU và masked max pooling.
3. BiLSTM-Attention: hidden size 64 mỗi hướng; attention học trọng số token, bỏ qua padding.
   Packed sequences giúp LSTM không xử lý phần padding như dữ liệu thật.
4. CNN-BiLSTM-Attention (Hybrid): ghép đặc trưng CNN và BiLSTM-Attention, dropout 0,35, linear 3 lớp.
   Hybrid là một mạng có nhiều nhánh, tự nó không phải ensemble các mô hình độc lập.
5. Soft-voting ensemble: kết hợp xác suất của ba mô hình được huấn luyện độc lập:
   `{', '.join(ensemble['members'])}`. Trọng số tương ứng: `{[round(w, 4) for w in ensemble['weights']]}`.
   Chọn trong 27 bộ trọng số dương đã định trước bằng macro F1 trên validation, không dùng test.

Với nhãn c, ensemble tính `P(c|x) = sum_m w_m * P_m(c|x)` và lấy lớp có xác suất lớn nhất.
Neural training dùng AdamW, learning rate {cfg['learning_rate']}, batch {cfg['batch_size']}, tối đa {cfg['epochs']} epoch,
early stopping patience {cfg['patience']} theo validation macro F1, gradient clipping 1,0 và weighted cross-entropy.
Trọng số lớp tính một lần từ train gốc và dùng chung cả khi tăng cường dữ liệu.
Checkpoint được khôi phục ở epoch có validation macro F1 cao nhất.

## 4. Data augmentation

Chỉ tăng cường phần train sau khi chia tập; validation và test không sinh thêm mẫu.
Dùng synonym replacement bằng từ điển nhỏ cố định và random deletion giới hạn vào các từ
`really`, `very`, `quite`, `just`. Mỗi mẫu sinh tối đa một biến thể cho từng phương pháp nếu đủ điều kiện.
Không thay/xóa từ phủ định hoặc các từ thuộc aspect; bảo toàn vị trí target sau biến đổi.
Loại biến thể trùng lặp và biến thể có câu trùng validation/test.
Số dòng theo phương pháp: `{audit['augmentation']}`.
Các mẫu và ID cha nằm trong `data/train_augmented.csv` để kiểm tra thủ công.

Các phép biến đổi là heuristic: bỏ từ tăng cường hoặc thay đồng nghĩa vẫn có thể đổi sắc thái,
do đó **không đảm bảo tăng chất lượng**. Trên lần chạy này augmentation {augmentation_comment} macro F1 test
{abs(augmentation_delta):.2f} điểm phần trăm so với cùng hybrid không augmentation.

## 5. Độ đo và kết quả thực nghiệm

Với mỗi lớp theo one-vs-rest: `Precision = TP/(TP+FP)`, `Recall = TP/(TP+FN)`,
`F1 = 2*Precision*Recall/(Precision+Recall)`; trường hợp mẫu số bằng 0 được gán 0.
Accuracy là tỷ lệ dự đoán đúng. Macro là trung bình đều ba lớp; weighted F1 có trọng số theo số mẫu.
ROC dùng xác suất của từng lớp, không dùng nhãn argmax; AUC báo cáo là macro one-vs-rest.
Macro F1 là tiêu chí chính vì dữ liệu lệch về positive. Các điểm dưới đây đều ở thang 0–1.

{markdown_table(joined)}

Mô hình được **chọn bằng validation** là **{selected}**:
test accuracy **{best['accuracy']:.4f}**, macro F1 **{best['f1_macro']:.4f}**,
macro ROC-AUC **{best['roc_auc_ovr_macro']:.4f}**.
Baseline ML tốt nhất theo validation là **{baseline}**;
chênh lệch macro F1 test của mô hình được chọn so với baseline đó là
**{100*(best['f1_macro']-scores.loc[baseline, 'f1_macro']):+.2f} điểm phần trăm**.
Không chọn lại mô hình dựa trên bảng test.

![So sánh mô hình](results/model_comparison.png)

### Ablation

{markdown_table(pd.DataFrame(effects), digits=2)}

Số dương là tăng, số âm là giảm. Đây là quan sát của một lần chạy với seed {cfg['seed']},
không phải bằng chứng có ý nghĩa thống kê hoặc bảo đảm phương pháp luôn tốt hơn.
Mô hình tốt hơn về accuracy có thể kém hơn về macro F1 nếu ưu tiên lớp positive quá nhiều.

### Kết quả theo lớp của mô hình được chọn

{markdown_table(per_class)}

![Confusion matrix](results/confusion_matrices.png)

![ROC one-vs-rest](results/roc_ovr.png)

![Lịch sử học](results/learning_curves.png)

## 6. Phân tích lỗi và hạn chế

Có **{len(mistakes)}/{audit['splits']['test']['rows']}** dự đoán sai ở mô hình được chọn.
Số lỗi theo hướng nhầm: {error_summary}.
Lớp neutral có recall **{per_class.loc[per_class['Nhãn'] == 'neutral', 'recall'].iloc[0]:.4f}**;
khả năng nhận diện lớp này còn hạn chế, nên cần xem macro F1 cùng accuracy.

Các ví dụ dự đoán sai thực tế (năm dòng đầu, không chọn lọc theo câu chuyện kết luận):

{chr(10).join(example_lines)}

Trong bài toán này cần kiểm tra các trường hợp nhiều aspect có cảm xúc đối lập, nhận xét ngầm,
phủ định và ranh giới neutral/positive/negative. Đây là các hướng phân tích; chưa tự gán nguyên nhân
cho từng lỗi nếu chưa đọc thủ công. Toàn bộ lỗi được lưu thành `results/errors_<model>.csv`.

Giới hạn: một seed, một miền restaurant, BERT đóng băng, có thể cắt mất ngữ cảnh xa của câu dài,
augmentation thủ công có thể làm đổi sắc thái. BERT pair chứa tên aspect nhưng chưa đánh dấu
vị trí xuất hiện trong câu; nếu cùng một từ xuất hiện nhiều lần với cảm xúc khác nhau, mô hình
có thể không phân biệt được occurrence. Không so trực tiếp điểm này với bài báo hoặc notebook
có cách chia train/validation và cách xử lý dữ liệu khác.
Hướng tiếp theo: chạy nhiều seed và báo cáo mean/std, fine-tune BERT có đánh dấu target span,
kiểm tra augmentation thủ công, thử back translation và đánh giá trên miền khác.

## 7. Tái lập và tệp nộp

Môi trường đã chạy: `{env}`. Cấu hình đầy đủ: `results/config.json`;
phiên bản BERT và thống kê token: `results/bert_features.json`;
lịch sử từng epoch: `results/history_*.csv`; log: `results/run.log`.
Đọc `README.md` để chạy lại hoặc dự đoán một cặp câu/khía cạnh mới.
Notebook `Lab02_Aspect_Sentiment.ipynb` trình bày quy trình và hiển thị kết quả đã chạy.
Không cần nộp thư mục cache vector BERT để đọc báo cáo; cache có thể tái tạo.

## 8. Tài liệu tham khảo

- Slide học phần: `../slide/Chapter3_Sentiment Analysis project implementation.pdf`, trang 11–21.
- [SemEval-2014 Task 4: mô tả aspect-based sentiment analysis](https://alt.qcri.org/semeval2014/task4/index.php).
- [BERT-base-uncased model card](https://huggingface.co/google-bert/bert-base-uncased).
- [PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.rnn.LSTM.html).
- [Scikit-learn: cross-validation và grouped splitting](https://scikit-learn.org/stable/modules/cross_validation.html).
"""
    if (ROOT / "cross_validation/results/completion.json").exists():
        text = ("> **Bản holdout để đối chiếu.** Đọc [BAO_CAO_CV.md](BAO_CAO_CV.md) "
                "và [Lab02_5Fold_CV.ipynb](Lab02_5Fold_CV.ipynb) để xem kết quả chạy đủ 5 fold.\n\n") + text
    (ROOT / "BAO_CAO.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    write_report()
