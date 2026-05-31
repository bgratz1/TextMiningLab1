# Poster Completion Checklist

**Status:** Poster structure complete with all missing methodological detail, theory citations, P-R analysis, true positive examples, caveats, and limitations added.

**Remaining work:** Fill in blanks, run predictions, export PDF.

---

## CRITICAL — Must Fill Before Submitting

### Student Names & Contributions

**Location:** Header (top center) + Division of Work section (bottom left)

- [ ] Replace `[Student 1] – [Student 5]` with real names (line ~105)
- [ ] Fill in detailed contributions for each student in Division of Work box (lines ~180–200)
  - Specific tasks coded (e.g., SVM pipeline setup, VADER experiments, BERT training)
  - Specific analyses done (e.g., sentiment error analysis, domain mismatch study)
  - Poster contributions (e.g., poster layout design, sentiment section writing)

### Test Set Predictions — Sentiment Analysis

**Location:** Sentiment section, "Test Set Results & Error Analysis"

- [ ] Run TF-IDF+SVM on `sentiment-topic-test.tsv` (18 sentences) and predict labels for all 18 sentences. Generate `classification_report(y_true, svm_predictions)` showing precision, recall, F1 per class. Replace placeholder `fig_sentiment_test_f1.png` with real results.
  
- [ ] Run VADER on same 18 sentences. Use `analyzer.polarity_scores()` to compute compound score, then threshold to classify. Generate `classification_report(y_true, vader_predictions)` and compare with SVM results.

- [ ] Fill in P-R tradeoff analysis: Which system is more conservative (high precision, low recall) vs aggressive (high recall, low precision)?

**Expected output file:** `visuals/fig_sentiment_test_f1_REAL.png` (rename and replace placeholder)

**Script to run:**
```python
# In final_project_Text_Mining.ipynb or separate script
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load test set
test_df = pd.read_csv("sentiment-topic-test.tsv", sep="\t")
X_test = test_df["sentence"].values
y_true = test_df["sentiment"].values

# SVM predictions (use trained model from airline tweets)
svm_preds = model.predict(X_test)  # your trained model
print(classification_report(y_true, svm_preds))

# VADER predictions
analyzer = SentimentIntensityAnalyzer()
vader_preds = [analyzer.polarity_scores(sent) for sent in X_test]
# ... thresholding logic
print(classification_report(y_true, vader_preds))

# Generate visualization — update generate_visuals.py with real data
```

### Test Set Predictions — NER

**Location:** NER section, "Test Set (NER-test.tsv)"

- [ ] Run LinearSVC on `NER-test.tsv` (15 sentences). Load test sentences, tokenize, extract {word, POS-tag} features per token, vectorize using same DictVectorizer as training, predict BIO labels. Generate `classification_report(y_true_ner, ner_predictions)` per entity class and replace placeholder `fig_ner_test_f1.png` with real results.

- [ ] Add observations about which entity types appear in test set (PERSON, ORG, LOC, WORK_OF_ART, etc.), whether WORK_OF_ART was predicted correctly or defaulted to O (expected: defaults to O/misclassifies since unseen in training), and include examples of correct and incorrect predictions.

**Expected output file:** `visuals/fig_ner_test_f1_REAL.png` (rename and replace placeholder)

**Script template:**
```python
# Load NER test set
ner_test_df = pd.read_csv("NER-test.tsv", sep="\t")

# Process per sentence
test_features = []
test_labels = []
for _, row in ner_test_df.iterrows():
    feature_dict = {"word": row["token"], "pos": row["POS"]}
    test_features.append(feature_dict)
    test_labels.append(row["BIO_NER_tag"])

# Vectorize and predict
X_test_ner = vec.transform(test_features)
ner_preds = ner_model.predict(X_test_ner)
print(classification_report(test_labels, ner_preds))
```

### GitHub Link

**Location:** References section (bottom right, "Source Code")

- [ ] Replace `[INSERT LINK]` with actual GitHub repository URL (e.g., `https://github.com/yourname/TextMiningLab1`)
- [ ] Ensure repo contains all notebooks (Final_Project_Text_Mining.ipynb, textminingproject.ipynb, etc.), generate_visuals.py script, visuals/ folder with all generated graphs, poster.html file, and optional test set predictions output

---

## OPTIONAL — Improves Quality but Not Required

### True Positive Examples

**Location:** Already added sample table in Sentiment section

- [ ] Verify or replace with actual examples from your test set predictions
  - Pick 1–2 sentences where BOTH systems got it right
  - Explain why (strong sentiment words, clear structure, etc.)
  - Add to NER section if time (1–2 correctly extracted entities)

### More Error Examples

**Location:** Sentiment & Topic error analysis sections

- [ ] Add 1–2 more concrete error examples per task
  - Show actual sentence from test set
  - Show what system predicted vs. ground truth
  - Briefly explain why error occurred
  - E.g., for topic: "Sentence: 'The atmosphere at the stadium was electric' (gold: sports, pred: electronics)"

### Confusion Matrix Details

**Location:** NER section

- [ ] Annotate confusion matrices with specific misclassification patterns (e.g., "I-ORG often confused with I-PER" or "B-WORK_OF_ART never predicted")

---

## EXPORT & SUBMIT

### Generate PDF

**Steps:**

1. Open `poster.html` in Google Chrome (full support for CSS Grid, images)
2. **File → Print** (or Cmd+P on Mac)
3. In Print dialog:
   - **Paper size:** A0 landscape
   - **Margins:** None / Minimal
   - **Destination:** Save as PDF
4. Save as: `final_project_poster.pdf`
5. Verify:
   - All visuals render correctly
   - Text not cut off
   - Colors print correctly (or acceptable in grayscale)

### Prepare Submission

- [ ] `final_project_poster.pdf` (main deliverable)
- [ ] Source code ZIP or GitHub link (in poster + separate submission if required)
- [ ] Ensure repo includes all notebooks, generate_visuals.py, visuals/ folder with all 13+ images, and any test set prediction outputs

### Final Review Against Rubric

**Data usage (15 pts):**
- [ ] Datasets described with sources, sizes, splits
- [ ] Collection strategy justified
- [ ] Class imbalance discussed
- [ ] Training vs test data characterized

**Methodology (15 pts):**
- [ ] Approaches named and motivated
- [ ] Preprocessing steps explicit
- [ ] Features/representations explained
- [ ] Hyperparameters listed with justification
- [ ] Theory cited (SVM convex opt, TF-IDF weighting, BERT bidirectional context, etc.)

**Results & Analysis (15 pts):**
- [ ] Quantitative metrics (F1, P, R, accuracy) per class
- [ ] Confusion matrices or equivalent
- [ ] Per-class error analysis (why does class X fail?)
- [ ] Qualitative examples (true positives, false positives, false negatives)
- [ ] P vs R tradeoff discussed

**Clarity & Division (5 pts):**
- [ ] Division of work clear (who coded what, who analyzed what, who did poster)
- [ ] All student names visible
- [ ] Contributions specific, not generic

---

## CHECKLIST SUMMARY

**Must Do (Blocking Submission):**
- [ ] Fill student names (5 students)
- [ ] Run sentiment test predictions → fill fig_sentiment_test_f1.png
- [ ] Run NER test predictions → fill fig_ner_test_f1.png
- [ ] Fill GitHub link
- [ ] Export to PDF

**Strongly Recommended (Improves Grade):**
- [ ] Add actual examples from your test predictions
- [ ] Verify all 13 visuals are present and readable in final PDF
- [ ] Test division of work section is clear and specific

**Total Estimated Time:**
Sentiment predictions: 15 min (run SVM + VADER on test set) · NER predictions: 15 min (run model + generate report) · Fill blanks: 10 min · PDF export & QA: 10 min · **Total: ~50 minutes**

---

## QUESTIONS TO ANSWER (Store in Poster or Notes)

Before filling blanks, answer these:
- **Sentiment test set:** Did you actually run SVM+VADER on sentiment-topic-test.tsv? If not, you need to do this now.
- **NER test set:** Did you run the trained LinearSVC on NER-test.tsv? If not, do this now.
- **Student names:** What are the 5 group members' names?
- **Contributions:** Who coded what? Who did analysis? Who did poster?
- **GitHub:** Is the repo set up? What's the URL?

---

**Document updated:** 2026-05-30  
**Version:** Poster structure complete, awaiting test set predictions and student info
