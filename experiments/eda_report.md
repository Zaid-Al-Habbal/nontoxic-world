# EDA Report

## General Information
- The dataset contains 159571 rows and 8 columns.
- The columns are: `id`, `comment_text`, `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.
- The training set has 143613 rows, and the test set has 15958 rows.

## Data Analysis

### Checking for Imbalance

- IR = Number of non-toxic comments / Number of toxic comments
- Number of non-toxic comments: 128975
- Number of toxic comments: 14638
- Imbalance Ratio (IR): 8.81 (significant imbalance)

- toxic percentage:
9.61%
- severe_toxic percentage:
1.01%
- obscene percentage:
5.30%
- threat percentage:
0.31%
- insult percentage:
4.95%
- identity_hate percentage:
0.89%

**Decision to make:**
- We may need to consider techniques to handle the imbalance in the dataset, such as class weighting in the loss function, tuning the decision threshold and PR-AUC as an evaluation metric.


### Co-occurrence between labels
#### High co-occurrence
- toxic and obscene
- toxic and insult
- obscene and insult
- toxic and severe_toxic (every severe toxic comment is also toxic)