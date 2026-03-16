# Nontoxic World Project Summary

This document summarized the actions we took in the project. The project consists of several stages, 
including EDA, data preprocessing, tokenization, prepare data for training, building models, training, and evaluation.

- [Nontoxic World Project Summary](#nontoxic-world-project-summary)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Data Preprocessing](#data-preprocessing)
      - [Wikimedia specific preprocessing:](#wikimedia-specific-preprocessing)
  - [Tokenization](#tokenization)
    - [BBPE Tokenizer](#bbpe-tokenizer)
  - [Prepare Data for Training](#prepare-data-for-training)
    - [Positive class weights](#positive-class-weights)
    - [Data Loaders](#data-loaders)
  - [Model Building, Training, and Evaluation](#model-building-training-and-evaluation)
    - [For Training:](#for-training)
    - [Model Architectures \& Results:](#model-architectures--results)
      - [1. TF-IDF + Logistic Regression](#1-tf-idf--logistic-regression)
        - [Architecture:](#architecture)
        - [Results on the test set:](#results-on-the-test-set)
      - [2. Stacked GRU Model:](#2-stacked-gru-model)
        - [Architecture:](#architecture-1)
        - [Results on the test set:](#results-on-the-test-set-1)
      - [3. Stacked BiGRU Model:](#3-stacked-bigru-model)
        - [Architecture:](#architecture-2)
        - [Results on the test set:](#results-on-the-test-set-2)
      - [4. Stacked GRU Model with Pretrained BERT Embeddings:](#4-stacked-gru-model-with-pretrained-bert-embeddings)
        - [Architecture:](#architecture-3)
        - [Results on the test set:](#results-on-the-test-set-3)
      - [4. Stacked BiGRU Model with Pretrained BERT Embeddings:](#4-stacked-bigru-model-with-pretrained-bert-embeddings)
        - [Architecture:](#architecture-4)
        - [Results on the test set:](#results-on-the-test-set-4)
      - [5. Stacked BiGRU + Dot-Product Attention with Pretrained BERT Embeddings (automatically fine-tuned using optuna)](#5-stacked-bigru--dot-product-attention-with-pretrained-bert-embeddings-automatically-fine-tuned-using-optuna)
        - [Architecture:](#architecture-5)
        - [Results on the test set:](#results-on-the-test-set-5)


## Exploratory Data Analysis (EDA)

During the EDA phase, we performed the following analyses on the dataset:
- Get to know the dataset (number of rows, columns, column names, etc.)
- Checking for Imbalance
- Co-occurrence between labels
- Comment Length Analysis (Character Length and Word Length)
- Wikimedia specific preprocessing analysis
- Vocabulary Size Estimation

## Data Preprocessing

We did the following:
- Handle None or empty comments by replacing them with "[UNK]"
- Lowercase all comments to ensure uniformity
- Replace URLs with a placeholder token "[UNK]"
- Normalize stretched characters (e.g. "loooool" -> "lool") 
- Normalize whitespace

#### Wikimedia specific preprocessing:
- Convert [[link|text]] -> text
- Remove templates {{...}}
- Remove bold/italic markup
- Remove headings

## Tokenization

We chose to use two tokenization methods for our experiments: BBPE (trained on our training set) and the pre-trained BERT tokenizer.

### BBPE Tokenizer

We trained a BBPE tokenizer on our training set using the `tokenizers` library. The tokenizer was trained with a vocabulary size of 30,000 and a minimum token frequency of 2. The resulting tokenizer was saved and used for tokenizing our dataset.

We enabled truncation to a maximum length of 256 tokens and padding to ensure that all sequences are of the same length.


## Prepare Data for Training

### Positive class weights

To address class imbalance in our dataset, we calculated class weights for the positive class for each label. The weights were computed based on the frequency of the positive class in the training data, and these weights were used during model training to give more importance to the positive class for each label.

### Data Loaders

- **We did tokenization on the dataset before creating the data loaders**. This allowed us to save the tokenized dataset and avoid redundant tokenization during each epoch of training.

- We used **dynamic padding** in our data loaders using `DataCollatorWithPadding` from the `transformers` library, which means that we only pad the sequences in each batch to the maximum length of the sequences in that batch. This approach helps to reduce the amount of padding and can improve training efficiency.

- Then we created data loaders for the training, validation, and test sets using the `DataLoader` class from PyTorch. We set the batch size to 256 and enabled shuffling for the training data loader to ensure that the model sees different batches of data in each epoch.

## Model Building, Training, and Evaluation

### For Training:

- We used AdamW as the optimizer with a learning rate of 0.001 and a weight decay of 0.01.
- We used Binary Cross-Entropy with positive class weights as the loss function to address class imbalance.
- We used **macro PR-AUC** as the main metric for evaluation during training because the dataset is imbalanced and we care about the overall performance, and we also monitored macro **F1-score**.
- We used performance scheduling to reduce the learning rate by a factor of 0.1 if the validation macro PR-AUC did not improve for 2 consecutive epochs.
- We saved the best model with its config to .pth files based on the highest validation macro PR-AUC.
- We logged training and validation metrics for each epoch to track experiments using ***W&B***

### Model Architectures & Results:

#### 1. TF-IDF + Logistic Regression

##### Architecture:
- TF-IDF vectorizer with a maximum vocabulary size of 30,000 and n-grams ranging from 1 to 3.
- Logistic Regression classifier with maximum iterations set to 500 and class weights set to 'balanced' to handle class imbalance.

##### Results on the test set:
- Macro PR-AUC: 0.62
- Macro F1-score: 0.54 @ threshold 0.5

#### 2. Stacked GRU Model:

##### Architecture:
- Embedding layer with a vocab size of 30,000 (trained BBPE tokenizer from scratch) and an embedding dimension of 256.
- 1 layer of GRU with a hidden dimension of 32 
- Output layer

##### Results on the test set:
- Macro PR-AUC: 0.55
- Macro F1-score: 0.54 @ threshold 0.23

#### 3. Stacked BiGRU Model:

##### Architecture:
- Embedding layer with a vocab size of 30,000 (trained BBPE tokenizer from scratch) and an embedding dimension of 256.
- 2 layer of BiGRU with a hidden dimension of 32 and dropout of 0.4
- Output layer

##### Results on the test set:
- Macro PR-AUC: 0.62
- Macro F1-score: 0.6 @ threshold 0.21

#### 4. Stacked GRU Model with Pretrained BERT Embeddings:

##### Architecture:
- Pretrained BERT Embeddings ("bert-base-uncased") freezed.
- 2 layer of GRU with a hidden dimension of 128 and dropout of 0.5
- Output layer

##### Results on the test set:
- Macro PR-AUC: 0.68
- Macro F1-score: 0.64 @ threshold 0.94


#### 4. Stacked BiGRU Model with Pretrained BERT Embeddings:

##### Architecture:
- Pretrained BERT Embeddings ("bert-base-uncased") freezed.
- 2 layer of BiGRU with a hidden dimension of 32 and dropout of 0.4
- Output layer

##### Results on the test set:
- Macro PR-AUC: 0.69
- Macro F1-score: 0.67 @ threshold 0.92

#### 5. Stacked BiGRU + Dot-Product Attention with Pretrained BERT Embeddings (automatically fine-tuned using optuna)

##### Architecture:
- Pretrained BERT Embeddings ("bert-base-uncased") freezed.
- 2 layer of BiGRU with a hidden dimension of 32 and dropout of 0.4
- Output layer

##### Results on the test set:
- Macro PR-AUC: 0.69
- Macro F1-score: 0.67 @ threshold 0.92