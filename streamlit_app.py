import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your dataset
df = pd.read_csv('finalSentimentdata2.csv')  # Replace 'your_dataset.csv' with the actual file path

# Data Preprocessing
le = LabelEncoder()
df['sentiment'] = le.fit_transform(df['sentiment'])  # Convert sentiments to numerical labels

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['sentiment']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Save the trained model
joblib.dump((classifier, le, vectorizer), 'sentiment_model.pkl')

# Application to predict sentiment for a given keyword
def predict_sentiment(keyword):
    # Load the trained model
    model, label_encoder, text_vectorizer = joblib.load('sentiment_model.pkl')

    # Predict sentiment for the keyword
    keyword_vectorized = text_vectorizer.transform([keyword])
    prediction = model.predict(keyword_vectorized)
    predicted_sentiment = label_encoder.inverse_transform(prediction)

    return predicted_sentiment[0]

# Streamlit Application
st.title("Twitter Sentiment Analysis during Lockdown in India")

# User input for keyword
user_input = st.text_input("Enter a keyword:")

# Perform sentiment prediction when the user clicks the button
if st.button("Predict Sentiment"):
    result = predict_sentiment(user_input)
    st.write(f"Sentiment for '{user_input}': {result}")