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
   pip install pandas nltk scikit-learn imbalanced-learn numpy
2. Clone the repository:
   git clone https://github.com/your_username/sentiment-analysis.git
3. Navigate to the project directory:
  cd sentiment-analysis
4. Run the script:
  python Clone the repository:

bash
Copy code
git clone https://github.com/your_username/sentiment-analysis.git
Navigate to the project directory:

bash
Copy code
cd sentiment-analysis
Run the script:

bash
Copy code
python sentiment_analysis.py
Results
The model achieved an accuracy of 0.69 on the test dataset.
Cross-validation results demonstrate the model's potential for generalization.
Future Improvements
Collect and incorporate more diverse data to enhance the model's performance.
Experiment with different classifiers and hyperparameter tuning for optimization.
Feel free to contribute and provide feedback!
## Results
- The model achieved an accuracy of 0.69 on the test dataset.
- Cross-validation results demonstrate the model's potential for generalization.
## Future Improvements
- Collect and incorporate more diverse data to enhance the model's performance.
- Experiment with different classifiers and hyperparameter tuning for optimization.
Feel free to contribute and provide feedback!
