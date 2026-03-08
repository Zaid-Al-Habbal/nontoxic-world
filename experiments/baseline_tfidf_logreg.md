# Baseline model

**Why TF-IDF + Logistic Regression?**
- Fast training and inference
- Interpretable feature weights
- Strong baseline for text classification
- Handles sparse high-dimensional data well

**Limitations:**
- Cannot capture word order beyond n-grams
- No semantic understanding
- Struggles with rare/misspelled words
- Fixed vocabulary after training

**Results**
- PR-AUC for toxic: 0.8603
- PR-AUC for severe_toxic: 0.4366
- PR-AUC for obscene: 0.8656
- PR-AUC for threat: 0.4232
- PR-AUC for insult: 0.7610
- PR-AUC for identity_hate: 0.4174
- Macro PR-AUC: 0.6274