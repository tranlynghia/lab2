"""Create the Vietnamese teaching notebook; outputs are added by execution."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent


def build():
    nb = nbf.v4.new_notebook()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb.cells = [
        md("""# Lab02 — Phân tích cảm xúc theo khía cạnh bằng Deep Learning

**Học phần:** Sentiment Analysis · **Nguồn bài tập:** Chapter 3, trang 21.

**Dữ liệu:** dùng nguyên `restaurant_aspect_train.csv` và `restaurant_aspect_test.csv` đã chuẩn hóa ở bài trước.
**Đơn vị dự đoán:** cặp *(câu, aspect_term)*. Không gộp nhãn để phân loại cả câu.

Bài làm gồm baseline ML; CNN/BiLSTM-Attention/hybrid; BERT embedding;
augmentation trên train; ensemble soft voting; Accuracy, Precision, Recall, F1, ROC-AUC;
confusion matrix và phân tích lỗi. Báo cáo hoàn chỉnh: [BAO_CAO.md](BAO_CAO.md).

Notebook đã lưu kết quả thực thi. Chọn kernel Python có dependencies trong `requirements.txt`.
Đặt `RUN_TRAINING = True` ở phần 7 nếu muốn huấn luyện lại toàn bộ."""),
        md("## 1. Khởi tạo môi trường"),
        code("""from pathlib import Path
import os, sys, json, inspect
os.environ.setdefault('USE_TF', '0')
import numpy as np
import pandas as pd
import torch
from IPython.display import display, Markdown, Image, Code

ROOT = Path.cwd()
if not (ROOT / 'run_experiment.py').exists():
    ROOT = ROOT / 'Lab02'
assert (ROOT / 'run_experiment.py').exists(), 'Mở notebook tại thư mục Lab02 hoặc thư mục cha.'
sys.path.insert(0, str(ROOT))
from data_utils import read_aspect_csv, LABELS, augment_train
from networks import SentimentNet
from run_experiment import Config, run, train_baselines, choose_ensemble
print('Thư mục:', ROOT)
print('PyTorch:', torch.__version__)
print('Thiết bị:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"""),
        md("""## 2. Dữ liệu aspect-level có sẵn

Mỗi dòng giữ `text`, `aspect_term`, `polarity` và vị trí `from/to`. Một câu có thể xuất hiện
nhiều lần với các khía cạnh/nhãn khác nhau. Ba lớp: negative, neutral, positive.
Các CSV nguồn đã loại conflict. Hai file nguồn trong Lab02 là bản sao nguyên byte từ bài trước."""),
        code("""raw_train = pd.read_csv(ROOT / 'data/restaurant_aspect_train.csv')
raw_test = pd.read_csv(ROOT / 'data/restaurant_aspect_test.csv')
print('Train gốc:', raw_train.shape, '| Test gốc:', raw_test.shape)
display(raw_train[['sentence_id', 'text', 'aspect_term', 'polarity']].head())
display(pd.DataFrame({'train': raw_train.polarity.value_counts(),
                      'test': raw_test.polarity.value_counts()}).reindex(LABELS))
mixed = raw_train.groupby('text').polarity.nunique()
example = mixed[mixed > 1].index[0]
display(raw_train.loc[raw_train.text == example, ['text', 'aspect_term', 'polarity']])"""),
        md("""## 3. Chia train/validation và tránh rò rỉ

Giữ nguyên 1.120 mẫu test. Loại một mẫu train có câu trùng test.
Lấy fold đầu của StratifiedGroupKFold 5 folds, seed 42. Group theo câu để các khía cạnh
của cùng câu không nằm ở cả train và validation; **nhãn vẫn theo từng khía cạnh**.
Đây là một holdout, chưa phải trung bình nhiều fold. Không dùng test để chọn mô hình."""),
        code("""train = pd.read_csv(ROOT / 'data/train.csv')
validation = pd.read_csv(ROOT / 'data/validation.csv')
test = pd.read_csv(ROOT / 'data/test.csv')
for a, b in [(train, validation), (train, test), (validation, test)]:
    assert set(a.group).isdisjoint(b.group)
audit = json.loads((ROOT / 'results/data_audit.json').read_text(encoding='utf-8'))
display(pd.DataFrame(audit['splits']).T)
print('Mẫu train loại do trùng test:', audit['removed_train_overlap'])
assert test[['text', 'aspect', 'label']].values.tolist() == raw_test[['text', 'aspect_term', 'polarity']].values.tolist()"""),
        md("""## 4. Baseline Machine Learning

TF-IDF unigram/bigram, có aspect và cửa sổ ngữ cảnh. Dùng Majority, Multinomial NB,
Logistic Regression và calibrated Linear SVM. Huấn luyện lại trên cùng train và đánh giá cùng test với DL.
SVM hiệu chỉnh xác suất bằng 3-fold theo nhóm trong train; vectorizer cũng fit trong từng fold.
Giữ từ phủ định, không xóa stopwords. Mã chạy baseline:"""),
        code("display(Code(inspect.getsource(train_baselines), language='python'))"),
        md("""## 5. Feature Engineering: BERT

Dùng pretrained `bert-base-uncased`, đóng băng encoder và lấy vector token tầng cuối (768 chiều).
Chuỗi vào là `[CLS] aspect [SEP] review [SEP]`. Chỉ cắt review khi vượt 96 token, giữ aspect.
Mạng phía sau học phép chiếu 768 → 96 chiều. Đây là feature extraction, không fine-tune BERT.

Không stemming hoặc loại dấu câu trước BERT. Embedding ngẫu nhiên dùng cùng WordPiece là mốc ablation.
[Model card BERT](https://huggingface.co/google-bert/bert-base-uncased)."""),
        code("""features_info = json.loads((ROOT / 'results/bert_features.json').read_text(encoding='utf-8'))
print('BERT:', features_info['model'], '| revision:', features_info['revision'])
print('Cấu trúc cặp:', features_info['pair_format'])
display(pd.DataFrame(features_info['splits']).T)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
sample = tokenizer('service', 'The food was excellent but the service was slow.')
print(tokenizer.convert_ids_to_tokens(sample['input_ids']))"""),
        md("""## 6. Data Augmentation

Sinh biến thể **chỉ từ train**: thay một từ bằng đồng nghĩa từ từ điển cố định;
hoặc xóa ngẫu nhiên một từ trong `really`, `very`, `quite`, `just` nếu có.
Giữ aspect, cập nhật offset và không xóa từ phủ định; loại bản trùng lặp/đụng validation hoặc test.
Đây là heuristic, không bảo đảm giữ trọn sắc thái. Thí nghiệm có/không augmentation sẽ kiểm chứng hiệu quả."""),
        code("""augmented = pd.read_csv(ROOT / 'data/train_augmented.csv')
display(augmented.augmentation.value_counts().rename('Số mẫu').to_frame())
generated = augmented[augmented.augmentation != 'original']
examples = generated.merge(train[['row_id', 'text']], left_on='parent_id', right_on='row_id', suffixes=('_aug', '_original'))
display(examples[['text_original', 'text_aug', 'aspect', 'label', 'augmentation']].head(8))
assert set(generated.group).isdisjoint(set(validation.group) | set(test.group))
assert all(r.text[r.start:r.end] == r.aspect for r in augmented.itertuples())"""),
        md("""## 7. Kiến trúc và huấn luyện

```text
(review, aspect) → WordPiece → BERT đóng băng → Linear(768, 96)
                                        ├→ CNN kernels 3/5/7 → masked max pool ┐
                                        └→ BiLSTM → masked attention         ├→ concat → dropout → 3 logits
                                                                             ┘
```

CNN và BiLSTM-Attention cũng được huấn luyện độc lập. `Random_Hybrid` thay BERT bằng
embedding khởi tạo ngẫu nhiên. Hybrid là một mạng ghép đặc trưng; soft voting ở phần sau mới là ensemble các mạng.

AdamW lr=0.001, batch=64, tối đa 20 epochs, patience=4 theo validation macro F1;
weighted cross-entropy từ train gốc, gradient clipping 1.0. Khôi phục checkpoint validation tốt nhất.
Mask và packed sequences ngăn padding tác động đến kết quả."""),
        code("""display(Code(inspect.getsource(SentimentNet), language='python'))
print('Cấu hình đã thực chạy:')
display(json.loads((ROOT / 'results/config.json').read_text()))"""),
        code("""RUN_TRAINING = False  # Đổi True để huấn luyện lại và ghi đè kết quả.
if RUN_TRAINING:
    run(Config())  # Máy mới: Config(local_files_only=False) để tải BERT.
else:
    assert (ROOT / 'results/metrics_test.csv').exists(), 'Chưa có kết quả: đặt RUN_TRAINING=True.'
    print('Đọc kết quả đã chạy; không huấn luyện lại trong lần mở notebook này.')"""),
        md("""## 8. Ensemble

Soft voting kết hợp xác suất BERT-CNN, BERT-BiLSTM-Attention và BERT-Hybrid-Augmented:
`P(c|x) = Σ w_m P_m(c|x)`. Tất cả trọng số dương và tổng bằng 1.
Chọn trọng số từ 27 tổ hợp cố định theo validation macro F1; test chỉ dùng báo cáo cuối."""),
        code("""ensemble = json.loads((ROOT / 'models/ensemble.json').read_text())
display(pd.DataFrame({'Mô hình': ensemble['members'], 'Trọng số': ensemble['weights']}))
print('Validation macro F1:', ensemble['validation_f1_macro'])
display(Code(inspect.getsource(choose_ensemble), language='python'))"""),
        md("""## 9. Đánh giá định lượng

Accuracy = số mẫu đúng / tổng mẫu. Precision = TP/(TP+FP); Recall = TP/(TP+FN);
F1 = 2PR/(P+R). Macro lấy trung bình đều ba lớp, weighted F1 theo support.
ROC dùng **xác suất từng lớp** theo one-vs-rest, AUC lấy macro.
Chọn mô hình bằng validation macro F1 vì positive chiếm đa số; không chọn theo test."""),
        code("""test_metrics = pd.read_csv(ROOT / 'results/metrics_test.csv')
val_metrics = pd.read_csv(ROOT / 'results/metrics_validation.csv')
comparison = val_metrics[['model', 'f1_macro']].rename(columns={'f1_macro': 'validation_f1'}).merge(test_metrics, on='model')
display(comparison.sort_values('validation_f1', ascending=False).round(4))
selected = json.loads((ROOT / 'models/selected_model.json').read_text())['model']
display(Markdown(f'**Mô hình được chọn theo validation:** `{selected}`'))
display(Image(filename=str(ROOT / 'results/model_comparison.png')))"""),
        md("## 10. Ablation: embedding, augmentation và ensemble"),
        code("""scores = test_metrics.set_index('model')
rows = []
for factor, before, after in [
    ('BERT embedding', 'Random_Hybrid', 'BERT_Hybrid'),
    ('Augmentation', 'BERT_Hybrid', 'BERT_Hybrid_Augmented'),
    ('Ensemble vs hybrid augmented', 'BERT_Hybrid_Augmented', 'SoftVoting_Ensemble')]:
    rows.append({'Thay đổi': factor, 'Trước': before, 'Sau': after,
                 'Delta accuracy (điểm %)': 100*(scores.loc[after, 'accuracy']-scores.loc[before, 'accuracy']),
                 'Delta macro F1 (điểm %)': 100*(scores.loc[after, 'f1_macro']-scores.loc[before, 'f1_macro'])})
display(pd.DataFrame(rows).round(2))
display(Markdown('Số âm là giảm chất lượng. Không khẳng định augmentation/ensemble luôn cải thiện. '
                 'Một seed và một holdout chưa đủ để kết luận ý nghĩa thống kê.'))
display(Image(filename=str(ROOT / 'results/learning_curves.png')))"""),
        md("## 11. Confusion matrix, ROC và phân tích lỗi"),
        code("""report = json.loads((ROOT / f'results/classification_report_{selected}.json').read_text())
display(pd.DataFrame({label: report[label] for label in LABELS}).T.round(4))
display(Image(filename=str(ROOT / 'results/confusion_matrices.png')))
display(Image(filename=str(ROOT / 'results/roc_ovr.png')))
errors = pd.read_csv(ROOT / f'results/errors_{selected}.csv')
print('Số dự đoán sai:', len(errors), '/', len(test))
display(errors[['text', 'aspect', 'label', 'predicted']].head(10))
display(errors.groupby(['label', 'predicted']).size().rename('Số lỗi').sort_values(ascending=False).to_frame())"""),
        md("""Kiểm tra thủ công lỗi ở các câu có nhiều aspect với cảm xúc đối lập, phủ định,
ý kiến ngầm hoặc sắc thái neutral. Không tự gán nguyên nhân lỗi nếu chưa đọc từng câu.
Giới hạn: BERT đóng băng, một seed, augmentation thủ công; đầu vào chưa đánh dấu vị trí aspect
nên có thể nhầm khi cùng tên aspect xuất hiện nhiều lần. Chưa kiểm chứng khả năng chuyển miền.

**Hướng cải tiến:** nhiều seed, fine-tune BERT có target markers, kiểm tra biến thể tăng cường,
back translation và đánh giá trên miền khác. Đánh giá mọi lựa chọn bằng validation trước khi dùng test."""),
        md("## 12. Nạp checkpoint để dự đoán một khía cạnh mới"),
        code("""from predict import predict
demo = predict('The food was excellent but the service was slow.', 'service')
display(demo)"""),
        md("""## 13. Kết luận và tài liệu

Các bảng trên là kết quả thực nghiệm, không phải số liệu kỳ vọng. Mọi phương pháp đều đánh giá
trên cùng 1.120 mẫu khía cạnh; test không tham gia huấn luyện, early stopping hay chọn trọng số ensemble.

- [Báo cáo đầy đủ và nhận xét từ số liệu](BAO_CAO.md).
- [Cách chạy và cấu trúc bài làm](README.md).
- Slide Chapter 3, trang 11–21, đặc biệt Exercise trang 21.
- [SemEval-2014 Task 4](https://alt.qcri.org/semeval2014/task4/index.php).
- [BERT-base-uncased](https://huggingface.co/google-bert/bert-base-uncased).
- [PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.rnn.LSTM.html).
- [Scikit-learn grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)."""),
    ]
    nb.metadata = dict(kernelspec=dict(display_name="Python 3 (py310)", language="python", name="python3"),
                       language_info=dict(name="python", version="3.10"))
    if (ROOT / "cross_validation/results/completion.json").exists():
        nb.cells.insert(0, md("**Bản holdout để đối chiếu.** Kết quả chạy đủ 5 fold nằm trong "
                              "[Lab02_5Fold_CV.ipynb](Lab02_5Fold_CV.ipynb) và "
                              "[BAO_CAO_CV.md](BAO_CAO_CV.md)."))
    nbf.validate(nb)
    path = ROOT / "Lab02_Aspect_Sentiment.ipynb"
    nbf.write(nb, path)
    print(path)


if __name__ == "__main__":
    build()
