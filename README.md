# Hate-Speech-Classification

## 1. Problem Definition
**Objective:** To build a model that can automatically detect hate speech in text data (tweets).

**Challenges:**
- Handling imbalanced data
- Understanding context in text
- Dealing with noise like slang and abbreviations

---

## 2. Dataset & Preprocessing
**Data Source:** A dataset of tweets labeled into 3 categories:
- **Hate Speech (Class 0)**
- **Offensive Language (Class 1)**
- **Neutral Speech (Class 2)**

**Preprocessing Steps:**
1. Convert text to lowercase.
2. Remove special characters, numbers, and punctuations.
3. Tokenization (splitting text into words).
4. Stopword removal (removing common words like "the", "is", etc.).
5. Lemmatization (converting words to base form, e.g., "running" → "run").
6. Vectorization using **TF-IDF** (to convert text into numerical format).

---

## 3. Models Used
Your project explores multiple models to classify hate speech:

### (A) Logistic Regression
- A simple yet effective linear model.
- Used as a baseline model.
- Works well when the dataset is not too large.

### (B) Support Vector Machine (SVM)
- Good for text classification since it works well with high-dimensional data.
- Maximizes the margin between different classes, making it robust for hate speech detection.
- Performed the best in terms of **precision & recall**.

### (C) Random Forest
- An ensemble learning model that combines multiple decision trees.
- Helps reduce overfitting compared to single decision trees.
- Works well on large datasets with imbalanced classes.

---

## 4. Performance Evaluation
Each model was evaluated based on:
- **Accuracy:** How many predictions were correct?
- **Precision:** Of the predicted hate speech, how many were actually hate speech?
- **Recall:** How well the model identified actual hate speech?
- **F1-score:** A balance between precision and recall.
- **Confusion Matrix:** To analyze misclassifications.

---
