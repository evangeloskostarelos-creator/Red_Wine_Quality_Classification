# Red Wine Quality Classification

A comprehensive machine learning project for predicting red wine quality based on physiochemical properties using PyCaret and ensemble methods.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [File Structure](#file-structure)
- [Key Findings](#key-findings)

## Overview

This project implements a machine learning classification model to predict red wine quality (rated 1-10) based on 12 physiochemical properties. The notebook demonstrates a complete data science workflow from exploratory data analysis through model development, evaluation, and deployment.

**Key Highlights:**
- Comprehensive EDA with visualizations
- Strategic feature engineering (10 new features)
- Ensemble model blending for robust predictions
- Focus on F1 score and Recall metrics for imbalanced classes
- Professional documentation and analysis

## Dataset

**Source:** UCI Machine Learning Repository  
**Dataset:** Red Wine Quality (winequality-red.csv)  
**Samples:** 1,599 red wine samples  
**Features:** 12 physiochemical properties

### Features

1. Fixed acidity
2. Volatile acidity
3. Citric acid
4. Residual sugar
5. Chlorides
6. Free sulfur dioxide
7. Total sulfur dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol
12. Quality (target variable, 1-10 scale)

### Class Distribution

The dataset exhibits **severe class imbalance**:
- Quality 5: 681 samples (42.6%) - Most common
- Quality 6: 638 samples (39.9%) - Second most common
- Quality 7: 199 samples (12.4%) - Moderate
- Quality 4: 53 samples (3.3%) - Underrepresented
- Quality 8: 18 samples (1.1%) - Severely underrepresented
- Quality 3: 10 samples (0.6%) - Extremely rare

## Features

### 1. Exploratory Data Analysis (EDA)
- Dataset overview and statistical summaries
- Feature relationship visualization (scatter matrix)
- Distribution analysis with boxplots
- Class imbalance identification

### 2. Feature Engineering
- **Ratio Features** (4 features):
  - Fixed to volatile acidity ratio
  - Citric to volatile acidity ratio
  - Free to total sulfur dioxide ratio
  - Alcohol to density ratio

- **Polynomial Features** (3 features):
  - Alcohol squared
  - Volatile acidity squared
  - Sulphates squared

- **Log Transformations** (3 features):
  - Log(1+x) for residual sugar
  - Log(1+x) for chlorides
  - Log(1+x) for total sulfur dioxide

**Total:** 12 original + 10 engineered = 22 features

### 3. Model Development
- PyCaret-based automated ML workflow
- Model comparison across multiple algorithms
- Ensemble blending of top 3 models by Recall
- Comprehensive model evaluation

### 4. Evaluation Metrics
- Confusion Matrix
- ROC-AUC Curves
- Classification Report (Precision, Recall, F1-score)
- Precision-Recall Curves
- Learning Curves

## Installation

### Prerequisites

- Python 3.7 - 3.11 (PyCaret compatibility)
- Jupyter Notebook or JupyterLab

### Required Packages

The notebook automatically installs required packages, but you can also install them manually:

```bash
pip install pandas numpy seaborn matplotlib pycaret
```

Or install from requirements:

```bash
pip install -r requirements.txt
```

**Note:** PyCaret requires Python 3.7-3.11. Python 3.12+ is not currently supported.

## Usage

### Running the Notebook

1. **Clone or download** the repository
2. **Ensure** `winequality-red.csv` is in the same directory as the notebook
3. **Open** `wine_classification_red.ipynb` in Jupyter Notebook/Lab
4. **Run all cells** sequentially (Cell → Run All)

### Notebook Structure

The notebook is organized into 5 main sections:

1. **Setup and Installation** - Package installation and imports
2. **Data Loading and EDA** - Data exploration and visualization
3. **Data Preprocessing and Feature Engineering** - Feature creation and transformation
4. **Model Development** - Model training, comparison, and evaluation
5. **Conclusions** - Summary and recommendations

### Expected Outputs

- Model comparison tables
- Evaluation visualizations (confusion matrix, ROC curves, etc.)
- Holdout set predictions
- Saved model pipeline (`red_wine_quality_pipeline.pkl`)

## Methodology

### Metric Selection Strategy

**Why Recall?**

- **Class Imbalance**: Most wines are rated 5 or 6, creating severe imbalance
- **Recall**: Ensures the model can identify all quality classes, including rare ones (3, 4, 8)

### Model Selection Process

1. **Recall-Based Selection**: Select top 3 models by Recall for ensemble
2. **Ensemble Blending**: Blend models to leverage strengths of multiple algorithms
3. **Evaluation**: Comprehensive evaluation across all quality classes

### Feature Engineering Rationale

- **Ratio Features**: Capture domain-relevant relationships (e.g., acidity balance)
- **Polynomial Features**: Enable non-linear pattern detection
- **Log Transformations**: Normalize right-skewed distributions

## Results

### Model Performance

The final blended ensemble model provides:
- Robust predictions for common quality classes (5, 6, 7)
- Moderate performance on rare classes (3, 4, 8) given severe class imbalance
- Improved generalization through ensemble blending

### Key Insights

- **Class Imbalance Impact**: Classes 3 and 8 have insufficient samples (<20 each) for reliable classification
- **Feature Engineering**: Expanded feature set from 12 to 22 features
- **Ensemble Benefits**: Blending multiple models improves robustness and generalization

## File Structure

```
.
├── wine_classification_red.ipynb    # Main notebook
├── winequality-red.csv              # Dataset
├── red_wine_quality_pipeline.pkl   # Saved model (generated after running)
└── README.md                        # This file
```

## Key Findings

### Dataset Characteristics
- **1,599 samples** with 12 original features
- **No missing values** - complete dataset
- **Severe class imbalance** - quality 3 (0.6%) and 8 (1.1%) are extremely rare

### Feature Engineering Impact
- Successfully expanded feature set to 22 features
- Ratio features capture domain relationships
- Log transformations normalize skewed distributions
- Polynomial features enable non-linear pattern detection

### Model Performance
- Ensemble blending improves robustness
- Recall metric prioritizes all classes, not just majority
- Model struggles with rare classes (3, 8) due to insufficient samples

