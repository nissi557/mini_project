# Import necessary libraries
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request

# Load the preprocessed dataset
df_merge = pd.concat([df_fake, df_true])
tfidf_vectorizer = TfidfVectorizer()
text = df_merge['text'].values
tfidf_vectorizer.fit(text)
text_vectorizer = tfidf_vectorizer.transform(text)

# Train a Logistic Regression model
x = text_vectorizer
y = df_merge["class"].values
log_reg = LogisticRegression()
log_reg.fit(x, y)

# Create a Flask web app
app = Flask(__name__)

# Define a function to predict the class of news
def predict_news(news_text):
    stemmed_text = stemming(news_text)
    text_vector = tfidf_vectorizer.transform([stemmed_text])
    prediction = log_reg.predict(text_vector)
    return "Real" if prediction == 1 else "Fake"

# Define a route for the home page
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        news_text = request.form.get("news_text")
        prediction = predict_news(news_text)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
