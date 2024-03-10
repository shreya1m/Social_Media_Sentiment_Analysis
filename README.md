# Social_Media_Sentiment_Analysis

# Sentiment Analysis using Random Forest Classifier

## Overview

This project aims to perform sentiment analysis on textual data using a Random Forest Classifier. The dataset comprises social media text, and the sentiment labels are categorized as Positive, Negative, and Neutral.

## Project Structure

- **Data Cleaning and Preprocessing**: 
  - Removed irrelevant columns.
  - Cleaned and processed text data using NLTK for tokenization, stemming, and removing stopwords.
  - Applied Sentiment Intensity Analyzer to categorize sentiments.

- **Vectorization**:
  - Used TF-IDF vectorization to convert text data into numerical features.

- **Modeling**:
  - Employed a Random Forest Classifier.
  - Addressed class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).

- **Evaluation**:
  - Evaluated the model using accuracy, confusion matrix, precision, and recall.
  - Conducted cross-validation to assess generalizability.

## Dependencies

- pandas
- nltk
- sklearn
- imbalanced-learn
- numpy

## Usage

1. Install dependencies:
   ```bash
   pip install pandas nltk scikit-learn imbalanced-learn numpy
