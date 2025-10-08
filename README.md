# AI vs Human Text Classification

A comprehensive machine learning project that classifies text as either AI-generated or human-written using multiple classification algorithms with explainable AI features.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Features](#features)
- [Results](#results)
- [Dependencies](#dependencies)

## 🎯 Overview

This project implements and compares multiple machine learning classifiers to distinguish between AI-generated and human-written text. The system uses natural language processing techniques and provides model interpretability through LIME (Local Interpretable Model-agnostic Explanations).

### Key Objectives
- Build robust text classification models
- Compare performance across different algorithms
- Provide explainable predictions using LIME
- Handle imbalanced datasets
- Deploy ensemble methods for improved accuracy

## 📊 Dataset

### Data Sources
- **AI-Generated Text**: 8 CSV files (`urlsf_subset00-06, 09.csv`) - 29,395 samples
- **Human-Written Text**: 8 CSV files (`urlsf_subset00-06, 09.csv`) - 29,142 samples
- **Total Dataset**: 58,537 samples after preprocessing

### Labels
- `0`: Human-written text
- `1`: AI-generated text

### Data Preprocessing
1. Concatenation of multiple CSV files
2. Removal of carriage returns (`\r`) and newlines (`\n`)
3. Duplicate removal
4. Dataset shuffling
5. Train-test split: 70-30

### Data Statistics
- **Training samples**: 40,975
- **Testing samples**: 17,562
- **Label distribution**: Nearly balanced (~50% each)

## 🤖 Models Implemented

### 1. AdaBoost Classifier
- **Base Estimator**: Decision Tree (max_depth=1)
- **N_estimators**: 50
- **F1-Score**: 0.88
- **Accuracy**: 88%

### 2. LightGBM (LGBM)
- **N_estimators**: 100
- **Learning Rate**: 0.1
- **F1-Score**: 0.92
- **Accuracy**: 92%
- **Best performing model**

### 3. Random Forest
- **N_estimators**: 100
- **F1-Score**: 0.85
- **Accuracy**: 84%

### 4. Hard Voting Classifier
- **Ensemble**: Naive Bayes + Random Forest + SVM
- **Voting**: Hard (Majority)
- **F1-Score**: 0.85
- **Accuracy**: 84%

### 5. Soft Voting Classifier
- **Ensemble**: Naive Bayes + Random Forest + SVM
- **Voting**: Soft (Probability-based)
- **Performance**: Similar to Hard Voting

### 6. Logistic Regression
- **Max iterations**: 1000
- **F1-Score**: 0.90
- **Accuracy**: 90%
- **Second-best performer**

## 🚀 Usage

### 1. Data Preparation

```python
import pandas as pd

# Load AI-generated text
ai_df = pd.read_csv('data/AI/urlsf_subset00.csv')

# Load human-written text
human_df = pd.read_csv('data/Human/urlsf_subset00.csv')

# Concatenate and preprocess
merged_df = pd.concat([ai_df, human_df])
```

### 2. Train a Model

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from lightgbm import LGBMClassifier

# Create pipeline
pipeline = Pipeline([
    ('count_vectorizer', CountVectorizer()),
    ('tfidf_transformer', TfidfTransformer()),
    ('lgbm', LGBMClassifier(n_estimators=100, learning_rate=0.1))
])

# Train
model = pipeline.fit(X_train, y_train)
```

### 3. Make Predictions

```python
# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Custom text prediction
custom_text = ["Your text here..."]
prediction = model.predict(custom_text)
```

### 4. Explain Predictions with LIME

```python
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=["Human", "AI"])
exp = explainer.explain_instance(
    text_instance, 
    model.predict_proba, 
    num_features=10
)
exp.show_in_notebook()
```

## 📈 Model Performance

### Test Set Performance (17,562 samples)

| Model | Accuracy | F1-Score | TPR | TNR | FPR | FNR |
|-------|----------|----------|-----|-----|-----|-----|
| **LightGBM** | **92%** | **0.92** | 0.93 | 0.90 | 0.10 | 0.07 |
| **Logistic Regression** | **90%** | **0.90** | 0.92 | 0.88 | 0.12 | 0.08 |
| **AdaBoost** | 88% | 0.88 | 0.90 | 0.86 | 0.14 | 0.10 |
| **Random Forest** | 84% | 0.85 | 0.91 | 0.77 | 0.23 | 0.09 |
| **Hard Voting** | 84% | 0.85 | 0.91 | 0.77 | 0.23 | 0.09 |

### Custom Test Set Performance (50 samples)

| Model | Accuracy | F1-Score | Correctly Labeled |
|-------|----------|----------|-------------------|
| **Logistic Regression** | **80%** | **0.83** | 40/50 |
| **LightGBM** | 74% | 0.60 | 37/50 |
| **AdaBoost** | 68% | 0.73 | 34/50 |
| **Random Forest** | 74% | 0.75 | 37/50 |

### Key Metrics Explained
- **TPR (True Positive Rate)**: Correctly identified AI-generated text
- **TNR (True Negative Rate)**: Correctly identified human-written text
- **FPR (False Positive Rate)**: Human text incorrectly classified as AI
- **FNR (False Negative Rate)**: AI text incorrectly classified as human

## ✨ Features

### 1. Exploratory Data Analysis
- Word frequency analysis (Top 50 words)
- Bag-of-Words visualization
- Label distribution analysis
- Text length statistics

### 2. Feature Engineering
- **CountVectorizer**: Converts text to token counts
- **TF-IDF Transformer**: Normalizes token frequencies
- **N-grams**: Unigrams and bigrams (1,2)
- **Stop words removal**: English stop words filtered

### 3. Model Interpretability (LIME)
- Feature importance visualization
- Word-level contribution analysis
- Prediction confidence scoring
- Least confident predictions identification

### 4. Model Serialization
All trained models are saved using `cloudpickle`:
```python
import cloudpickle
with open('model.pkl', 'wb') as f:
    cloudpickle.dump(model, f)
```

### 5. Comprehensive Evaluation
- Classification reports
- Confusion matrices
- ROC curves
- Custom test set evaluation

## 🎯 Results

### Key Findings

1. **Best Model**: LightGBM achieves the highest performance (92% accuracy, 0.92 F1-score)
2. **Feature Importance**: Words like "significant", "to", "which", "of" are strong indicators
3. **AI Detection**: Models show high TPR (0.90-0.93), effectively catching AI-generated content
4. **False Positives**: FPR ranges from 0.10-0.23, indicating some human text misclassified as AI

### Important Words for Classification

**AI-Generated Text Indicators**:
- "significant", "which", "to", "including", "making"
- Formal language patterns
- Structured sentence composition

**Human-Written Text Indicators**:
- "of", "in", "million", "said", "you"
- Conversational tone
- Varied sentence structures

### Least Confident Predictions
Models identify edge cases with ~50% confidence, including:
- News articles with mixed writing styles
- Technical documentation
- Formal reports
- Opinion pieces

## 🔍 Model Analysis

### Confusion Matrix Interpretation

**LightGBM Confusion Matrix** (Best Model):
```
                Predicted
                Human    AI
Actual Human     7857    873
Actual AI         619   8213
```

- **True Negatives**: 7,857 (Human correctly identified)
- **False Positives**: 873 (Human misclassified as AI)
- **False Negatives**: 619 (AI misclassified as Human)
- **True Positives**: 8,213 (AI correctly identified)

## 🛠️ Advanced Usage

### Batch Prediction

```python
# Load saved model
import cloudpickle
with open('lightgbm_model.pkl', 'rb') as f:
    model = cloudpickle.load(f)

# Batch predict
texts = ["Text 1...", "Text 2...", "Text 3..."]
predictions = model.predict(texts)
probabilities = model.predict_proba(texts)
```

### Confidence Thresholding

```python
# Get predictions with confidence threshold
probabilities = model.predict_proba(X_test)
confidences = np.max(probabilities, axis=1)

# Filter by confidence
high_confidence = confidences > 0.8
reliable_predictions = predictions[high_confidence]
```
## 👥 Authors
Aryan Kakran


**Last Updated**: 07 October 2025
