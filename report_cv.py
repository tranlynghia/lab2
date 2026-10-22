"""Generate the report, figures and notebook for the full 5-fold experiment."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from data_utils import LABELS
from write_report import markdown_table

ROOT = Path(__file__).resolve().parent
CV = ROOT / "cross_validation"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def draw_cv_figures(summary, fold_table, selected):
    ordered = summary.sort_values("f1_macro_mean")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(ordered.model, ordered.f1_macro_mean, xerr=ordered.f1_macro_std,
            color="#246da6", capsize=4, alpha=.9)
    ax.set(xlim=(0, 1), xlabel="Outer-fold macro F1 (mean +/- sample SD, 5 folds)",
           title="Full grouped 5-fold CV | inner holdout for model tuning")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(CV / "results/cv_comparison.png", dpi=180)
    plt.close(fig)
    pivot = fold_table.pivot(index="model", columns="fold", values="f1_macro").loc[summary.model]
    fig, ax = plt.subplots(figsize=(10, 6))
    shown = ax.imshow(pivot.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(pivot)):
        for j in range(5):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                    color="white" if value > .6 else "black")
    ax.set(xticks=range(5), xticklabels=[f"Fold {i}" for i in range(1, 6)],
           yticks=range(len(pivot)), yticklabels=pivot.index,
           title="Outer-fold macro F1: each row was held out exactly once")
    fig.colorbar(shown, ax=ax, label="Macro F1")
    fig.tight_layout()
    fig.savefig(CV / "results/fold_scores.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, split, title in zip(axes, ["oof", "test"], ["Out-of-fold predictions", "Official test: five-model average"]):
        predictions = pd.read_csv(CV / "results" / f"predictions_{split}_{selected}.csv")
        cm = confusion_matrix(predictions.label, predictions.predicted, labels=LABELS)
        ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black")
        ax.set(xticks=range(3), xticklabels=LABELS, yticks=range(3), yticklabels=LABELS,
               xlabel="Predicted", ylabel="True", title=title)
    fig.suptitle(selected)
    fig.tight_layout()
    fig.savefig(CV / "results/selected_confusion.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, split in zip(axes, ["oof", "test"]):
        predictions = pd.read_csv(CV / "results" / f"predictions_{split}_{selected}.csv")
        for label in LABELS:
            fpr, tpr, _ = roc_curve((predictions.label == label).astype(int), predictions[f"p_{label}"])
            ax.plot(fpr, tpr, label=label)
        ax.plot([0, 1], [0, 1], "k--", alpha=.4)
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"{split.upper()} ROC (one-vs-rest)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(CV / "results/selected_roc.png", dpi=180)
    plt.close(fig)


def build_cv_notebook():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb = nbf.v4.new_notebook(cells=[
        md("""# Lab02 — Đánh giá đầy đủ 5-fold cross-validation

**Aspect-level, 3 lớp**, dùng hai CSV chuẩn hóa có sẵn. Đây là lần chạy mới đủ 5 fold,
tách biệt kết quả holdout trong notebook cũ. Mỗi mô hình DL được khởi tạo và huấn luyện lại ở từng fold.
Tổng cộng 10 phương pháp × 5 fold = 50 lượt đánh giá outer; 25 mạng DL và 20 baseline được fit.

Mỗi fold có **inner-fit → inner-validation → outer-evaluation**. Inner-validation chọn epoch/trọng số ensemble;
outer-evaluation chỉ chấm điểm. Test chính thức giữ riêng. Báo cáo: [BAO_CAO_CV.md](BAO_CAO_CV.md)."""),
        code("""from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT = Path.cwd()
if not (ROOT / 'run_cross_validation.py').exists():
    ROOT = ROOT / 'Lab02'
sys.path.insert(0, str(ROOT))
CV = ROOT / 'cross_validation'
def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))
completion = read_json(CV / 'results/completion.json')
assert completion['completed_folds'] == 5
display(completion)"""),
        md("""## 1. Dữ liệu và cách chia

Train nguồn có 3.602 mẫu khía cạnh. Loại một mẫu trùng câu với test khỏi phần development;
3.601 mẫu còn lại tham gia đủ 5 fold. Test có 1.120 mẫu, giữ nguyên.
`StratifiedGroupKFold(5)` chia theo nhóm câu chuẩn hóa, không gộp nhãn của các khía cạnh.
Bên trong 4/5 phần outer-train, lấy một grouped holdout khoảng 20% làm inner-validation.
Không refit trên toàn outer-train sau khi chọn epoch: baseline và DL dùng cùng inner-fit.
Vì vậy mỗi mạng học khoảng 64% development; khoảng 16% để chọn và 20% để chấm outer."""),
        code("""audit = read_json(CV / 'results/data_audit.json')
display(pd.DataFrame(audit['folds']))
assignments = pd.read_csv(CV / 'data/fold_assignments.csv')
assert len(assignments) == 3601
assert assignments.row_id.is_unique
assert assignments.groupby('group').outer_fold.nunique().max() == 1
display(pd.crosstab(assignments.outer_fold, assignments.label))
for fold in range(1, 6):
    base = CV / f'fold_{fold:02d}' / 'data'
    fit = pd.read_csv(base / 'inner_fit.csv')
    val = pd.read_csv(base / 'inner_validation.csv')
    outer = pd.read_csv(base / 'outer_evaluation.csv')
    for a, b in [(fit, val), (fit, outer), (val, outer)]:
        assert set(a.group).isdisjoint(b.group)
print('Đã kiểm tra các nhóm câu không giao nhau trong mọi fold.')"""),
        md("""## 2. Quy trình huấn luyện

Giữ kiến trúc và hyperparameter của Lab02: TF-IDF + baseline; random-embedding Hybrid;
BERT đóng băng + CNN/BiLSTM-Attention/Hybrid; Hybrid tăng cường dữ liệu; soft voting.
Augmentation chỉ tạo từ inner-fit của từng fold. TF-IDF chỉ fit trong phần train tương ứng.
BERT là bộ biến đổi cố định đã pretrained nên có thể dùng chung cache vector; không cập nhật từ outer/test.
Checkpoint và trọng số ensemble chọn bằng inner-validation, không bằng outer fold.
Thư mục `fold_01` đến `fold_05` lưu đầy đủ checkpoint, phân chia dữ liệu và lịch sử học."""),
        code("""display(read_json(CV / 'results/config.json'))
rows = []
for fold in range(1, 6):
    folder = CV / f'fold_{fold:02d}'
    aug = read_json(folder / 'results/augmentation.json')['counts']
    ensemble = read_json(folder / 'models/ensemble.json')
    rows.append({'fold': fold, **aug, 'ensemble_weights': ensemble['weights']})
display(pd.DataFrame(rows))"""),
        code("""RUN_TRAINING = False
if RUN_TRAINING:
    from run_cross_validation import run_cv
    from run_experiment import Config
    run_cv(Config())  # Tự tiếp tục fold đã hoàn thành nếu cấu hình và mã nguồn khớp.
else:
    print('Đọc kết quả thực chạy đủ 5 fold, không huấn luyện lại.')"""),
        md("""## 3. Điểm trung bình và độ lệch chuẩn của 5 outer folds

Mỗi độ đo tính riêng ở mỗi outer fold. Mean là trung bình đều 5 điểm;
SD là độ lệch chuẩn mẫu (`ddof=1`). Đây là độ dao động giữa các fold, không phải khoảng tin cậy.
Macro-F1 là tiêu chí chọn mô hình. Mỗi dòng development được dự đoán outer đúng một lần."""),
        code("""summary = pd.read_csv(CV / 'results/metrics_cv_summary.csv')
metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted', 'roc_auc_ovr_macro']
formatted = pd.DataFrame({'model': summary.model})
for metric in metrics:
    formatted[metric + ' mean ± SD (%)'] = [f'{100*m:.2f} ± {100*s:.2f}' for m,s in zip(summary[metric+'_mean'], summary[metric+'_std'])]
display(formatted)
display(Image(filename=str(CV / 'results/cv_comparison.png')))
fold_metrics = pd.read_csv(CV / 'results/metrics_folds.csv')
display(fold_metrics.pivot(index='model', columns='fold', values='f1_macro').round(4))
display(Image(filename=str(CV / 'results/fold_scores.png')))"""),
        md("""## 4. OOF và lựa chọn mô hình

OOF nối dự đoán outer của 5 fold theo đúng mẫu gốc. Metrics OOF được tính trên toàn bộ 3.601 dự đoán;
không đồng nhất với mean của 5 metrics vì kích thước fold khác nhau và F1 không tuyến tính.
Chọn phương pháp theo mean outer Macro-F1, không chọn theo test."""),
        code("""oof_metrics = pd.read_csv(CV / 'results/metrics_oof.csv')
display(oof_metrics.round(4))
selected = read_json(CV / 'models/selected_model.json')
display(selected)
assert selected['model'] == summary.sort_values('f1_macro_mean', ascending=False).iloc[0].model
for name in summary.model:
    oof = pd.read_csv(CV / f'results/predictions_oof_{name}.csv')
    assert len(oof) == 3601 and oof.row_id.is_unique
print('Mọi phương pháp đều có đủ 3.601 dự đoán OOF.')"""),
        md("""## 5. Test chính thức sau khi chọn mô hình

Với mỗi phương pháp, lấy trung bình xác suất của 5 estimator tương ứng rồi đánh giá trên cùng
1.120 mẫu test. Đây là **ensemble qua 5 fold**, không phải chia test thành 5 fold.
Nếu phương pháp là SoftVoting, mỗi fold đã kết hợp ba mạng; sau đó mới trung bình qua fold.
Test này đã được dùng trong thí nghiệm holdout trước, không phải một test mới chưa từng quan sát.
Trong lượt CV này, nhãn test không tham gia training, early stopping hoặc chọn trọng số."""),
        code("""test_metrics = pd.read_csv(CV / 'results/metrics_test.csv')
display(test_metrics.round(4))
display(Markdown(f"**Phương pháp đã chọn bằng CV:** `{selected['model']}`"))
display(test_metrics[test_metrics.model == selected['model']].round(4))
display(Image(filename=str(CV / 'results/selected_confusion.png')))
display(Image(filename=str(CV / 'results/selected_roc.png')))"""),
        md("## 6. Ablation và phân tích lỗi"),
        code("""effects = []
for factor, before, after in [
    ('BERT embedding', 'Random_Hybrid', 'BERT_Hybrid'),
    ('Augmentation', 'BERT_Hybrid', 'BERT_Hybrid_Augmented'),
    ('Soft voting vs augmented Hybrid', 'BERT_Hybrid_Augmented', 'SoftVoting_Ensemble')]:
    a = fold_metrics[fold_metrics.model == before].set_index('fold').f1_macro
    b = fold_metrics[fold_metrics.model == after].set_index('fold').f1_macro
    delta = 100*(b-a)
    effects.append({'Thay đổi': factor, 'Delta F1 trung bình (điểm %)': delta.mean(),
                    'SD delta': delta.std(ddof=1), 'Số fold cải thiện': int((delta>0).sum())})
display(pd.DataFrame(effects).round(3))
errors = pd.read_csv(CV / f"results/errors_{selected['model']}.csv")
display(errors[['text', 'aspect', 'label', 'predicted']].head(10))
print('Số lỗi test:', len(errors))"""),
        md("""## 7. Hạn chế và tái lập

Đủ 5 outer folds nhưng chỉ một seed cho cách chia. Các fold có phần train chồng lấp;
SD không phải sai số của 5 thí nghiệm hoàn toàn độc lập. Chọn mô hình từ bảng CV cũng có thiên lệch lựa chọn;
test cố định giúp đối chiếu nhưng đã được xem ở bài holdout trước. Chưa fine-tune BERT, chưa thêm lớp conflict.
Không suy ra ý nghĩa thống kê chỉ từ chênh lệch mean. Random-vs-BERT còn khác số tham số.

Chạy: `python run_cross_validation.py`; máy chưa có BERT cache: thêm `--allow-download`.
Kiểm tra: `python -m unittest discover -s tests -v`.
Đọc [BAO_CAO_CV.md](BAO_CAO_CV.md) và [README.md](README.md) để xem chi tiết tệp kết quả.
Notebook holdout cũ được giữ lại để đối chiếu, không dùng nhầm số liệu cũ làm kết quả 5-fold."""),
    ])
    nb.metadata = dict(kernelspec=dict(name="python3", display_name="Python 3 (py310)", language="python"),
                       language_info=dict(name="python", version="3.10"))
    nbf.validate(nb)
    nbf.write(nb, ROOT / "Lab02_5Fold_CV.ipynb")


def write_cv_report():
    summary = pd.read_csv(CV / "results/metrics_cv_summary.csv")
    folds = pd.read_csv(CV / "results/metrics_folds.csv")
    test = pd.read_csv(CV / "results/metrics_test.csv")
    selection = read_json(CV / "models/selected_model.json")
    audit = read_json(CV / "results/data_audit.json")
    config = read_json(CV / "results/config.json")
    selected = selection["model"]
    draw_cv_figures(summary, folds, selected)
    formatted = pd.DataFrame({"Mô hình": summary.model})
    for metric, label in [("accuracy", "Accuracy"), ("precision_macro", "Precision macro"),
                          ("recall_macro", "Recall macro"), ("f1_macro", "F1 macro"),
                          ("f1_weighted", "F1 weighted"), ("roc_auc_ovr_macro", "AUC OvR macro")]:
        formatted[label] = [f"{100*m:.2f} ± {100*s:.2f}" for m, s in zip(summary[f"{metric}_mean"], summary[f"{metric}_std"])]
    selected_test = test.set_index("model").loc[selected]
    pivot = folds.pivot(index="model", columns="fold", values="f1_macro").loc[summary.model].reset_index()
    pivot.columns = [str(column) for column in pivot.columns]
    effects = []
    for factor, before, after in [("BERT embedding", "Random_Hybrid", "BERT_Hybrid"),
                                  ("Augmentation", "BERT_Hybrid", "BERT_Hybrid_Augmented"),
                                  ("Ensemble vs Hybrid augmented", "BERT_Hybrid_Augmented", "SoftVoting_Ensemble")]:
        delta = (folds[folds.model == after].set_index("fold").f1_macro -
                 folds[folds.model == before].set_index("fold").f1_macro)*100
        effects.append({"Thay đổi": factor, "Delta F1 mean (điểm %)": delta.mean(),
                        "SD delta": delta.std(ddof=1), "Fold tăng": int((delta > 0).sum())})
    text = f"""# Lab02 — Kết quả đầy đủ 5-fold cross-validation

## 1. Phạm vi

Đã huấn luyện lại và chấm **10 phương pháp trên đủ 5 outer folds**, tổng cộng **50 lượt đánh giá**.
Gồm 25 mạng DL được fit từ đầu, 20 baseline được fit và 5 soft-voting ensemble.
Đây là kết quả mới, tách biệt bài holdout trước trong `BAO_CAO.md`.
Bài toán giữ nguyên **aspect polarity classification 3 lớp**: negative, neutral, positive.
Đầu vào là `(review, aspect)`; không gộp nhãn thành phân loại cả câu và chưa bổ sung lớp conflict.

## 2. Dữ liệu và thiết kế đánh giá

Dùng nguyên CSV chuẩn hóa: 3.602 train và 1.120 test. Bỏ {audit['removed_overlap']} mẫu train trùng câu với test
khỏi phần development, còn **{audit['development']} mẫu**. Không sửa file nguồn và giữ nguyên 1.120 test.
Mã SHA-256 được lưu trong `cross_validation/results/data_audit.json`.

Outer splitter: `StratifiedGroupKFold(5, shuffle=True, random_state={config['seed']})`.
Group theo câu chuẩn hóa để các khía cạnh của cùng câu luôn cùng fold.
Trong mỗi outer-train, chia thêm grouped holdout bằng 5-way splitter với seed `42 + fold`;
lấy một phần làm inner-validation, còn lại làm inner-fit.

{markdown_table(pd.DataFrame(audit['folds']))}

- **Inner-fit:** cập nhật tham số, fit TF-IDF, tạo augmentation, tính class weights.
- **Inner-validation:** chọn epoch tốt nhất và trọng số ensemble.
- **Outer-evaluation:** chỉ chấm điểm sau khi lựa chọn nội bộ hoàn tất.
- **Official test:** đánh giá cuối sau khi đã chọn phương pháp bằng mean CV Macro-F1.

Không refit trên toàn outer-train sau khi chọn epoch. Baseline và DL cùng dùng inner-fit,
tức khoảng 64% development; inner-validation khoảng 16%; outer-evaluation khoảng 20%.
Thiết kế này tránh dùng chính fold đang chấm để chọn epoch/trọng số, nhưng có ít mẫu cập nhật trọng số
hơn cách dùng toàn 4/5 development để train. Vì vậy không so trực tiếp mức điểm với holdout cũ như
thể mọi điều kiện huấn luyện đều giống nhau.
Mỗi mẫu development có **đúng một dự đoán out-of-fold (OOF)** cho mỗi phương pháp.

## 3. Mô hình và huấn luyện

Giữ kiến trúc Lab02: Majority; TF-IDF + NB/LR/SVM; Random Hybrid; BERT-CNN;
BERT-BiLSTM-Attention; BERT-Hybrid; BERT-Hybrid-Augmented; SoftVoting Ensemble.
BERT `bert-base-uncased` đóng băng, cặp `[CLS] aspect [SEP] review [SEP]`, tối đa 96 token.
Vector BERT có thể cache chung vì encoder không học tham số hay thống kê từ tập outer/test.
Mỗi head DL được khởi tạo lại ở mỗi fold. Các mô hình khác nhau giữ cùng seed khởi tạo {config['seed']}.

AdamW lr={config['learning_rate']}, batch={config['batch_size']}, tối đa {config['epochs']} epoch,
patience={config['patience']}, dropout 0,35, gradient clipping 1,0, weighted cross-entropy.
Early stopping theo **inner-validation Macro-F1**. SVM dùng 3-fold grouped calibration bên trong inner-fit.
Augmentation chỉ từ inner-fit: thay đồng nghĩa và restricted deletion, bảo vệ aspect/phủ định.
Trong mỗi fold, soft voting của CNN/BiLSTM/Hybrid-Augmented chọn trọng số bằng inner-validation.
Không dùng nhãn outer hoặc test để chọn trọng số đó.

## 4. Kết quả cross-validation: mean ± SD

Tất cả ô dưới đây ở **thang phần trăm**. Mean là trung bình đều năm độ đo outer;
SD là **độ lệch chuẩn mẫu, ddof=1**, không phải khoảng tin cậy.
Macro Precision/Recall/F1 lấy trung bình đều ba lớp; AUC dùng xác suất one-vs-rest.

{markdown_table(formatted)}

![CV mean và SD](cross_validation/results/cv_comparison.png)

Macro-F1 theo từng fold (thang 0–1):

{markdown_table(pivot)}

![Điểm từng fold](cross_validation/results/fold_scores.png)

Phương pháp được chọn là **{selected}**, với CV Macro-F1
**{100*selection['f1_macro_mean']:.2f}% ± {100*selection['f1_macro_std']:.2f} điểm phần trăm**.
Lựa chọn được lưu trước khi tính bảng metrics test.
`metrics_oof.csv` tính metrics trên toàn bộ dự đoán OOF đã nối; F1 OOF không nhất thiết bằng mean F1
do F1 phi tuyến và kích thước các fold không hoàn toàn bằng nhau.

## 5. Test chính thức và cách sử dụng mô hình cuối

Với mỗi phương pháp, lấy **trung bình xác suất của 5 estimator ứng với 5 fold**,
rồi dự đoán toàn bộ 1.120 mẫu test. Đây là ensemble qua fold, không phải chia test thành năm phần.
Riêng SoftVoting đã kết hợp ba mạng trong mỗi fold trước khi lấy trung bình qua fold.
Không chọn fold tốt nhất theo test và không chọn lại phương pháp theo bảng test.

{markdown_table(test)}

Phương pháp được chọn bằng CV đạt test Accuracy **{100*selected_test.accuracy:.2f}%**,
Macro-F1 **{100*selected_test.f1_macro:.2f}%**, AUC **{selected_test.roc_auc_ovr_macro:.4f}**.
`metrics_same_test_by_fold.csv` lưu điểm của từng fold-model trên cùng test để kiểm tra,
**không phải 5-fold CV trên test**, không dùng các dòng đó như năm tập test độc lập.
Test cố định này đã xuất hiện ở thí nghiệm holdout trước; không phải một tập test mới chưa từng được xem.

![Confusion matrix](cross_validation/results/selected_confusion.png)
![ROC](cross_validation/results/selected_roc.png)

## 6. Ablation và giới hạn

Chênh lệch Macro-F1 ghép cặp trên cùng outer fold:

{markdown_table(pd.DataFrame(effects), digits=3)}

Các fold có phần train chồng lấp; SD mô tả biến động theo cách chia, không chứng minh ý nghĩa thống kê.
Mới có một seed outer split, không phải repeated CV. Việc chọn phương pháp có mean cao nhất trên CV
cũng có thể gây thiên lệch lựa chọn. Chưa fine-tune BERT, chưa bổ sung conflict, chưa đánh dấu vị trí target.
So sánh Random/BERT còn khác số tham số; augmentation có thể đổi sắc thái và không đảm bảo tăng điểm.
Đọc `errors_<model>.csv` và `classification_report_<model>.json` để xem các lỗi/lớp yếu.

## 7. Tệp và tái lập

- `Lab02_5Fold_CV.ipynb`: notebook mới trình bày và hiển thị kết quả 5-fold.
- `run_cross_validation.py`: toàn bộ quy trình, tự resume fold hoàn thành khi cấu hình/mã nguồn/dữ liệu khớp.
- `cross_validation/fold_01` đến `fold_05`: các tập dữ liệu, 9 checkpoint/baseline, trọng số ensemble và lịch sử.
- `cross_validation/results/metrics_folds.csv`: 50 dòng độ đo outer.
- `cross_validation/results/metrics_cv_summary.csv`: mean/std của mọi độ đo cho 10 phương pháp.
- `cross_validation/results/predictions_oof_*.csv`: đủ 3.601 dự đoán OOF cho mỗi phương pháp.
- `cross_validation/results/metrics_test.csv`: điểm test sau khi trung bình xác suất qua fold.
- `cross_validation/models/selected_model.json`: phương pháp được chọn bằng CV.
- `cross_validation/results/run.log`, `environment.json`, `run_signature.json`: log, phiên bản và mã kiểm tra tái lập.

Chạy `python run_cross_validation.py` trong Lab02; thêm `--allow-download` nếu chưa có BERT cache.
Giữ cache vector bằng `--keep-cache`; mặc định xóa cache vector tạm sau khi dùng, giữ checkpoint.
Chạy `python -m unittest discover -s tests -v` để kiểm tra.
Mã nguồn mô hình và dữ liệu tham chiếu trong `README.md`, `BAO_CAO.md` và notebook holdout trước.
"""
    (ROOT / "BAO_CAO_CV.md").write_text(text, encoding="utf-8")
    build_cv_notebook()


if __name__ == "__main__":
    write_cv_report()
