from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re
from nltk.corpus import stopwords

app = Flask(__name__)

# Load the trained model and vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    article = request.form['article']
    # Preprocess the article
    article_vectorized = vectorizer.transform([article])
    prediction = model.predict(article_vectorized)
    result = "Real" if prediction[0] == 1 else "Fake"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)