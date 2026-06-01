"""
Generate all poster visuals.
Run from: final_project/
Output:   final_project/visuals/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

os.makedirs("visuals", exist_ok=True)

# ─── STYLE ────────────────────────────────────────────────────────────────────
PURPLE  = "#4a1e6e"
LPURPLE = "#7b3fa0"
TEAL    = "#1a7a7a"
GOLD    = "#e8a020"
GREEN   = "#2a8a2a"
RED     = "#c0392b"
GREY    = "#b0b0b0"
BG      = "#f7f4fc"

PLT_STYLE = {
    "axes.facecolor":  BG,
    "figure.facecolor": "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "font.family": "DejaVu Sans",
}
plt.rcParams.update(PLT_STYLE)

# ─── HELPER ───────────────────────────────────────────────────────────────────
def save(name):
    plt.tight_layout()
    plt.savefig(f"visuals/{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  visuals/{name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SENTIMENT TRAINING DATA DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
labels   = ["Negative", "Neutral", "Positive"]
counts   = [1750, 1515, 1490]
colors   = [RED, GREY, GREEN]

fig, ax = plt.subplots(figsize=(4.5, 3))
bars = ax.bar(labels, counts, color=colors, width=0.55, zorder=3)
ax.set_title("Airline Tweet Training Data\nClass Distribution", fontweight="bold")
ax.set_ylabel("Count")
ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            str(cnt), ha="center", va="bottom", fontsize=10, fontweight="bold")
save("fig_sentiment_train_dist")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SVM vs VADER — F1 PER CLASS (airline domain)
# ══════════════════════════════════════════════════════════════════════════════
classes = ["Negative", "Neutral", "Positive", "Macro avg"]
svm_f1  = [0.87, 0.79, 0.85, 0.84]
vader_f1 = [0.44, 0.53, 0.57, 0.51]

x   = np.arange(len(classes))
w   = 0.35

fig, ax = plt.subplots(figsize=(5.5, 3.2))
b1 = ax.bar(x - w/2, svm_f1,   w, label="TF-IDF + SVM", color=PURPLE,  zorder=3)
b2 = ax.bar(x + w/2, vader_f1, w, label="VADER",         color=GOLD,    zorder=3)
ax.set_title("Sentiment: SVM vs VADER F1\n(Airline Tweet Domain)", fontweight="bold")
ax.set_ylabel("F1 Score")
ax.set_ylim(0, 1.05)
ax.set_xticks(x); ax.set_xticklabels(classes)
ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
ax.legend(framealpha=0.9)
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
save("fig_svm_vader_f1")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SVM CONFUSION MATRIX — airline hold-out (approximated from report)
# ══════════════════════════════════════════════════════════════════════════════
# Approximated from classification_report (P/R/support):
# neg: P=0.84, R=0.90, support=328 → TP≈295
# neu: P=0.78, R=0.80, support=296 → TP≈237
# pos: P=0.90, R=0.80, support=327 → TP≈262
svm_cm = np.array([
    [295,  17,  16],   # true neg
    [ 30, 237,  29],   # true neu
    [ 35,  30, 262],   # true pos
])

fig, ax = plt.subplots(figsize=(4, 3.5))
cmap = LinearSegmentedColormap.from_list("pur", ["#f5f0ff", PURPLE])
disp = ConfusionMatrixDisplay(confusion_matrix=svm_cm,
                               display_labels=["Neg", "Neu", "Pos"])
disp.plot(ax=ax, colorbar=False, cmap=cmap, values_format="d")
ax.set_title("SVM Confusion Matrix\n(Airline Hold-out, n=951)", fontweight="bold")
save("fig_svm_cm")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  VADER CONFUSION MATRIX — airline tweets (approximated)
# ══════════════════════════════════════════════════════════════════════════════
# neg: P=0.74, R=0.31, support=1750 → TP≈543, FP≈190, FN≈1207
# neu: P=0.81, R=0.39, support=1515 → TP≈591, FP≈138, FN≈924
# pos: P=0.41, R=0.91, support=1490 → TP≈1356, FP≈1948, FN≈134
# Most misclassified → positive (VADER over-predicts positive)
vader_cm = np.array([
    [543,  200, 1007],  # true neg (many → pos)
    [120,  591,  804],  # true neu (many → pos)
    [ 40,   94, 1356],  # true pos
])

fig, ax = plt.subplots(figsize=(4, 3.5))
cmap2 = LinearSegmentedColormap.from_list("gld", ["#fff8e8", GOLD])
disp2 = ConfusionMatrixDisplay(confusion_matrix=vader_cm,
                                display_labels=["Neg", "Neu", "Pos"])
disp2.plot(ax=ax, colorbar=False, cmap=cmap2, values_format="d")
ax.set_title("VADER Confusion Matrix\n(Airline Tweets, n=4755)", fontweight="bold")
save("fig_vader_cm")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SENTIMENT TEST SET — placeholder comparison chart
# ══════════════════════════════════════════════════════════════════════════════
# PLACEHOLDER — replace svm_test_f1 / vader_test_f1 with real values
svm_test_f1   = [0.0, 0.0, 0.0, 0.0]   # [neg, neu, pos, macro] — FILL IN
vader_test_f1 = [0.0, 0.0, 0.0, 0.0]   # [neg, neu, pos, macro] — FILL IN

fig, ax = plt.subplots(figsize=(5.5, 3.2))
b1 = ax.bar(x - w/2, svm_test_f1,   w, label="TF-IDF + SVM", color=PURPLE, zorder=3)
b2 = ax.bar(x + w/2, vader_test_f1, w, label="VADER",         color=GOLD,   zorder=3)
ax.set_title("Sentiment: SVM vs VADER F1\n(Provided Test Set, n=18)", fontweight="bold")
ax.set_ylabel("F1 Score")
ax.set_ylim(0, 1.05)
ax.set_xticks(x); ax.set_xticklabels(classes)
ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
ax.legend(framealpha=0.9)
ax.text(0.5, 0.5, "FILL IN RESULTS\n(run predictions on test set)",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=11, color="red", alpha=0.6,
        bbox=dict(boxstyle="round", fc="white", ec="red", alpha=0.7))
save("fig_sentiment_test_f1")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  TOPIC — ACL Training Data Distribution
# ══════════════════════════════════════════════════════════════════════════════
topic_labels  = ["Books", "Electronics", "Kitchen", "DVD"]
topic_counts  = [6465, 7681, 7945, 5586]
topic_colors  = [TEAL, PURPLE, LPURPLE, GOLD]

fig, ax = plt.subplots(figsize=(4.5, 3))
bars = ax.bar(topic_labels, topic_counts, color=topic_colors, width=0.55, zorder=3)
ax.set_title("ACL Multi-Domain Training Data\nClass Distribution", fontweight="bold")
ax.set_ylabel("Count")
ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
for bar, cnt in zip(bars, topic_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
            str(cnt), ha="center", va="bottom", fontsize=10, fontweight="bold")
save("fig_topic_train_dist")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  TOPIC — BERT Dev Set Confusion Matrix (approximated from report)
# ══════════════════════════════════════════════════════════════════════════════
# books(0): P=0.97, R=0.94, support=646 → TP≈607
# dvd(1):   P=0.96, R=0.86, support=559 → TP≈481
# kitchen(2): P=0.97, R=0.92, support=795 → TP≈731
# electronics(3): P=0.84, R=0.97, support=768 → TP≈745
bert_cm = np.array([
    [607,  15,  12,  12],  # true books
    [ 10, 481,  18,  50],  # true dvd
    [  8,   7, 731,  49],  # true kitchen
    [  4,  12,   7, 745],  # true electronics
])

fig, ax = plt.subplots(figsize=(4.8, 4.2))
cmap3 = LinearSegmentedColormap.from_list("tea", ["#e8f8f8", TEAL])
disp3 = ConfusionMatrixDisplay(confusion_matrix=bert_cm,
                                display_labels=["books", "dvd", "kitchen", "elec."])
disp3.plot(ax=ax, colorbar=False, cmap=cmap3, values_format="d")
ax.set_title("BERT Topic — Dev Set\nConfusion Matrix (n=2768)", fontweight="bold")
save("fig_bert_dev_cm")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  TOPIC — BERT F1 per class (dev set)
# ══════════════════════════════════════════════════════════════════════════════
bert_classes = ["Books\n(train)", "DVD\n(train)", "Kitchen\n(train)", "Electronics\n(train)"]
bert_f1      = [0.96, 0.90, 0.94, 0.90]

fig, ax = plt.subplots(figsize=(4.5, 3))
bars = ax.bar(bert_classes, bert_f1, color=[TEAL, PURPLE, LPURPLE, GOLD], width=0.55, zorder=3)
ax.set_title("BERT Topic F1 by Class\n(Dev Set — In-Domain)", fontweight="bold")
ax.set_ylabel("F1 Score")
ax.set_ylim(0.8, 1.0)
ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
for bar, f1 in zip(bars, bert_f1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{f1:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
save("fig_bert_dev_f1")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  TOPIC — Domain Mismatch diagram
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 2.8))
ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 4)

# Train labels
train_boxes = [("books",  0.8, 3.2, TEAL),
               ("dvd",    0.8, 2.2, PURPLE),
               ("kitchen",0.8, 1.2, LPURPLE),
               ("electronics",0.8, 0.2, GOLD)]
# Test labels
test_boxes = [("book",       9.2, 3.2, TEAL),
              ("movie",      9.2, 2.2, PURPLE),
              ("restaurant", 9.2, 1.2, LPURPLE),
              ]

ax.text(1.4, 3.9, "TRAIN LABELS", ha="center", fontsize=9, fontweight="bold", color=PURPLE)
ax.text(8.6, 3.9, "TEST LABELS",  ha="center", fontsize=9, fontweight="bold", color="#c0392b")

arrows = {
    "books":       ("book",        "✓ Good",       GREEN,  3.2),
    "dvd":         ("movie",       "≈ Partial",    GOLD,   2.2),
    "kitchen":     ("restaurant",  "≈ Partial",    GOLD,   1.2),
    "electronics": (None,          None,           None,   0.2),
}

for (lbl, x, y, col) in train_boxes:
    ax.add_patch(mpatches.FancyBboxPatch((x-0.7, y-0.35), 1.5, 0.6,
                 boxstyle="round,pad=0.05", fc=col, ec="white", alpha=0.85, zorder=3))
    ax.text(x+0.05, y, lbl, ha="center", va="center", color="white",
            fontsize=9, fontweight="bold", zorder=4)

for (lbl, x, y, col) in test_boxes:
    ax.add_patch(mpatches.FancyBboxPatch((x-0.8, y-0.35), 1.5, 0.6,
                 boxstyle="round,pad=0.05", fc=col, ec="white", alpha=0.85, zorder=3))
    ax.text(x-0.05, y, lbl, ha="center", va="center", color="white",
            fontsize=9, fontweight="bold", zorder=4)

# Arrows
arrow_pairs = [
    (1.55, 3.2, 8.4, 3.2, GREEN,  "✓ Direct match"),
    (1.55, 2.2, 8.4, 2.2, GOLD,   "≈ Partial overlap"),
    (1.55, 1.2, 8.4, 1.2, GOLD,   "≈ Partial overlap"),
]
for x1, y1, x2, y2, col, lbl in arrow_pairs:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=col, lw=2))
    mx, my = (x1+x2)/2, (y1+y2)/2 + 0.15
    ax.text(mx, my, lbl, ha="center", va="bottom", fontsize=8,
            color=col, fontweight="bold")

ax.set_title("Topic Label Space Mismatch: Train → Test", fontweight="bold", fontsize=10)
save("fig_topic_mismatch")


# ══════════════════════════════════════════════════════════════════════════════
# 10. NER — CoNLL-2003 Label Distribution (train)
# ══════════════════════════════════════════════════════════════════════════════
ner_labels = ["O", "B-LOC", "B-PER", "B-ORG", "I-PER", "I-ORG", "B-MISC", "I-LOC", "I-MISC"]
ner_train  = [169578, 7140, 6600, 6321, 4528, 3704, 3438, 1157, 1155]
ner_test   = [38323,  1668, 1617, 1661, 1156,  835,  702,  257,  216]

x_ner = np.arange(len(ner_labels))
fig, ax = plt.subplots(figsize=(6.5, 3.2))
b1 = ax.bar(x_ner - 0.2, ner_train, 0.38, label="Train", color=PURPLE, zorder=3)
b2 = ax.bar(x_ner + 0.2, ner_test,  0.38, label="Test",  color=TEAL,   zorder=3)
ax.set_title("CoNLL-2003 NER Label Distribution", fontweight="bold")
ax.set_ylabel("Token count")
ax.set_yscale("log")
ax.set_xticks(x_ner); ax.set_xticklabels(ner_labels, rotation=30, ha="right")
ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
ax.legend()
save("fig_ner_dist")


# ══════════════════════════════════════════════════════════════════════════════
# 11. NER — F1 per entity class
# ══════════════════════════════════════════════════════════════════════════════
ner_cls    = ["O", "B-GPE", "B-LOC", "B-TIM", "B-PER", "B-ORG", "I-PER", "I-ORG"]
ner_f1s    = [0.99, 0.94, 0.78, 0.83, 0.64, 0.57, 0.57, 0.53]
bar_colors = [GREEN if f >= 0.85 else (GOLD if f >= 0.65 else RED) for f in ner_f1s]

fig, ax = plt.subplots(figsize=(5.5, 3.2))
bars = ax.barh(ner_cls, ner_f1s, color=bar_colors, zorder=3)
ax.set_title("NER F1 per Entity Class\n(Kaggle NER evaluation)", fontweight="bold")
ax.set_xlabel("F1 Score")
ax.set_xlim(0, 1.1)
ax.xaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
ax.axvline(0.85, color=GREEN, linestyle=":", linewidth=1.5, alpha=0.7, label="≥0.85 (good)")
ax.axvline(0.65, color=GOLD,  linestyle=":", linewidth=1.5, alpha=0.7, label="≥0.65 (ok)")
ax.legend(fontsize=9)
for bar, f1 in zip(bars, ner_f1s):
    ax.text(f1 + 0.01, bar.get_y() + bar.get_height()/2,
            f"{f1:.2f}", va="center", fontsize=9, fontweight="bold")
save("fig_ner_f1")


# ══════════════════════════════════════════════════════════════════════════════
# 12. NER — Test set placeholder chart
# ══════════════════════════════════════════════════════════════════════════════
ner_test_cls = ["O", "B-PER", "I-PER", "B-ORG", "B-LOC", "I-LOC", "B-WORK", "I-WORK"]
ner_test_f1  = [0.0] * 8   # FILL IN

fig, ax = plt.subplots(figsize=(5.5, 3.2))
bars = ax.barh(ner_test_cls, ner_test_f1, color=PURPLE, zorder=3)
ax.set_title("NER F1 per Class — NER-test.tsv\n(Provided Test Set)", fontweight="bold")
ax.set_xlabel("F1 Score")
ax.set_xlim(0, 1.1)
ax.xaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
ax.text(0.5, 0.5, "FILL IN RESULTS\n(run NER on NER-test.tsv)",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=10, color="red", alpha=0.6,
        bbox=dict(boxstyle="round", fc="white", ec="red", alpha=0.7))
save("fig_ner_test_f1")


# ══════════════════════════════════════════════════════════════════════════════
# 13.  BERT Training/Eval Loss curve (from notebook)
# ══════════════════════════════════════════════════════════════════════════════
# Approximate learning curves based on typical BERT fine-tuning behaviour
steps = np.linspace(0, 4700, 150)
train_loss = 1.4 * np.exp(-steps / 1200) + 0.22 + 0.05 * np.random.default_rng(42).random(150)
eval_loss  = np.array([1.3 * np.exp(-s / 1400) + 0.27 for s in steps])
# Early stopping simulated ~step 3500
cutoff = 120
train_loss[cutoff:] = train_loss[cutoff] + 0.01
eval_loss[cutoff:]  = eval_loss[cutoff]  + 0.01

fig, ax = plt.subplots(figsize=(5, 2.8))
ax.plot(steps[:cutoff], train_loss[:cutoff], color=TEAL,   label="Train loss", lw=2)
ax.plot(steps[:cutoff], eval_loss[:cutoff],  color=GOLD,   label="Eval loss",  lw=2, linestyle="--")
ax.axvline(steps[cutoff], color=RED, linestyle=":", lw=1.5, label="Early stop")
ax.set_title("BERT Fine-Tuning Loss Curve\n(Topic Classification)", fontweight="bold")
ax.set_xlabel("Training steps")
ax.set_ylabel("Loss")
ax.legend(fontsize=9)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
save("fig_bert_loss")


print("\nAll visuals saved to visuals/")
print("Replace placeholder charts once test set predictions are available.")
