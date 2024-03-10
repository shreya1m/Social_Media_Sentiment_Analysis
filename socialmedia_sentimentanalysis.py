# Importing Libraries
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import numpy as np
import string

df=pd.read_csv('social_text.csv')

df.info()

df.describe()

df.head()

# Dropping irrelavent columns
df.drop(columns=['Unnamed: 0.1'], inplace=True)
df.drop(columns=['Unnamed: 0'], inplace=True)

# Downloading NLTK resources (stopwords and punkt tokenizer)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

df.head(50)

# Checking the unique sentiments present in the 'Sentiment' target column
df["Sentiment"].nunique()

# Stripping extra spaces to ensure accurate sentiment count
df['Sentiment'] = df['Sentiment'].map(lambda x: x.strip())

# Creating copy of Dataframe
cdf=df.copy()

nltk.download('vader_lexicon')

# Create a SentimentIntensityAnalyzer instance
sia = SentimentIntensityAnalyzer()

# Get the sentiment column as a list to use with the SentimentIntensityAnalyzer object.
all_sentiment_words = df['Sentiment'].unique().tolist()

# Considering the existence of 191 different sentiments, which are essentially synonyms falling under the broader categories of positive, negative, and neutral,
# we aim to generalize them using a Sentiment Intensity Analyzer. This analyzer assigns a polarity score, aiding in classifying words as positive, negative, or neutral.
# Following the analysis, positive words will be placed in the 'pos_words' list, negative words in 'neg_words', and neutral words in 'neu_words'.
# Subsequently, we will replace all occurrences of positive words with the label 'Positive', negative words with 'Negative',
# and neutral words with 'Neutral sentiment', consolidating the sentiments into three distinct categories for improved clarity.

pos_words = [] # List for positive words
neg_words = [] # List for negative words
neu_words = [] # List for neutral words

for word in all_sentiment_words:

    # Get compound score of the word.
    score = sia.polarity_scores(word)['compound']
    if score >= 0.35:
        pos_words.append(word)
    elif score <= -0.35:
        neg_words.append(word)
    else:
        neu_words.append(word)

print(f'Positive words:\n{pos_words}\n\nNegative words:\n{neg_words}\n\nNeutral words:\n{neu_words}\n')
print(f'Pos len: {len(pos_words)}\nNeg len: {len(neg_words)}\nNeu len: {len(neu_words)}')

# Checking sentiment before
cdf['Sentiment'].nunique()

# Despite categorizing based on polarity scores, some words are misclassified in the neutral category.
# To enhance model accuracy, we are removing certain words from neutral that should be in positive or negative,
# and subsequently, adding them to the positive or negative category.

neg_words_to_move = ['Despair', 'Jealousy', 'Anxiety', 'Envious', 'Darkness', 'Desolation', 'Loss', 'Heartache']

pos_words_to_move = ['Fulfillment', 'Motivation', 'JoyfulReunion', 'Accomplishment', 'Wonderment', 'Enchantment',
                     'PlayfulJoy', 'FestiveJoy', 'Celebration']

# Loop for deleting from negative words(neg_words_to_move) from neutral list
for neg_word in neg_words_to_move:

    # While looping, get the index of the current word and delete it from neutral words list.
    index = neu_words.index(neg_word)
    del neu_words[index]

print(f'Neutral words:\n{neu_words}\n\nNeutral words length: {len(neu_words)}')

# Same process for positive words that are in the neu_words list
for pos_word in pos_words_to_move:
    index = neu_words.index(pos_word)
    del neu_words[index]

# Notice how the neu_words list gets smaller by the amount of words in both pos/neg_words_to_remove,to confirm things are working fine.
print(f'Neutral words:\n{neu_words}\n\nNeutral words length: {len(neu_words)}')

# Adding these pos and neg words to their appropriate pos_words and neg_words lists.
for word in pos_words_to_move:
    pos_words.append(word)

for word in neg_words_to_move:
    neg_words.append(word)

print(f'Positive words:\n{pos_words}\nPositive words length: {len(pos_words)}\n\n')
print(f'Negative words:\n{neg_words}\nNegative words length: {len(neg_words)}')

# Substituting pos_words with 'Positive', neg_words with 'Negative', and neu_words with 'Neutral' values in the 'Sentiment' target variable in the copied DataFrame cdf.

cdf['Sentiment'] = cdf['Sentiment'].map(lambda x: "Positive" if x in pos_words else x)
cdf['Sentiment'] = cdf['Sentiment'].map(lambda x: "Negative" if x in neg_words else x)
cdf['Sentiment'] = cdf['Sentiment'].map(lambda x: "Neutral" if x in neu_words else x)

# Tokenization

def clean_text(text):
    # Remove special characters, punctuation, and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Convert text to lowercase
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove extra whitespaces
    text = re.sub(' +', ' ', text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    # Reassemble the tokens into a single string
    text = ' '.join(tokens)


    return text

cdf['Processed_Text'] = cdf['Text'].apply(clean_text)

# Vectorization

# Create the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the documents
X = tfidf_vectorizer.fit_transform(cdf['Processed_Text'])

# Pritning  Vectorization Features

feature_names = tfidf_vectorizer.get_feature_names_out()
print("Feature Names:", feature_names)

# Assigning the 'Sentiment' column as the target variable y
y=cdf['Sentiment']
y

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Oversampling using SMOTE

# Create the SMOTE oversampler
smote = SMOTE(random_state=42)

# Fit and transform the training data using SMOTE
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Using Random Forest Classifier Model
model = RandomForestClassifier(n_estimators=150, class_weight='balanced',random_state=42)
model.fit(X_resampled, y_resampled)

#  Prediction using testing dataset
y_pred = model.predict(X_test)

# Comparing predicted values of the model when we pass testing values with the actual
# testing y values
print('Testing Sentiment Values: \n',y_test[20:50])
print('Model Predicted Values using X_test: \n' ,y_pred[20:50])

# Model Prediction on new Sentense

new_sentence = "It tastes so bitter"
processed_sentence = clean_text(new_sentence) # processing new_sentense using our function clean_text
vectorized_sentence = tfidf_vectorizer.transform([processed_sentence]) # vectorizing

# Make predictions
prediction = model.predict(vectorized_sentence)

print("Processed Sentence:", processed_sentence)
print("Predicted Class:", prediction)

# Drawing Confusion matrix for checking our model accuracy

conf_matrix = confusion_matrix(y_test, y_pred)

# Display the Confusion Matrix
print("Confusion Matrix:")
print(conf_matrix)

# Using precision , recall to check Accuracy

precision_positive = precision_score(y_test, y_pred, labels=[0], average='weighted', zero_division=1)
recall_positive = recall_score(y_test, y_pred, labels=[0], average='weighted', zero_division=1)

precision_negative = precision_score(y_test, y_pred, labels=[1], average='weighted', zero_division=1)
recall_negative = recall_score(y_test, y_pred, labels=[1], average='weighted', zero_division=1)

precision_neutral = precision_score(y_test, y_pred, labels=[2], average='weighted', zero_division=1)
recall_neutral = recall_score(y_test, y_pred, labels=[2], average='weighted', zero_division=1)

print(f"Precision for Class 0 (Positive): {precision_positive:.4f}")
print(f"Recall for Class 0 (Positive): {recall_positive:.4f}\n")

print(f"Precision for Class 1 (Negative): {precision_negative:.4f}")
print(f"Recall for Class 1 (Negative): {recall_negative:.4f}\n")

print(f"Precision for Class 2 (Neutral): {precision_neutral:.4f}")
print(f"Recall for Class 2 (Neutral): {recall_neutral:.4f}")

# Using cross-validation as well for checking our model accuracy

# Use StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and get accuracy scores
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

# Display the accuracy scores
print("Cross-Validation Accuracy Scores:", scores)

# Mean Accuracy score
print("Mean Accuracy:", np.mean(scores))

# Final testing using accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# The obtained accuracy of 0.69 is promising given the limited dataset.
# However, with additional data, the model's performance is likely to improve significantly.
# Despite the data constraints, various sampling techniques and checks were applied
# to enhance the accuracy, demonstrating the model's potential.
