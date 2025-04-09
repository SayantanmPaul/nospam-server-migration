from flask import Flask, request, jsonify
import string

import numpy as np
import pandas as pd
import pickle
import os

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask_cors import CORS

nltk.download('stopwords')

app = Flask(__name__)
CORS(app, resources={r"/predict-spam": {"origins": "*", "allow_headers": "*", "methods": ["POST"]}})

model_path = 'model.pkl'
vectorizer_path = 'vectorizer.pkl'

vectorizer = CountVectorizer()
clf = RandomForestClassifier(n_jobs=-1)
ps = PorterStemmer()

# preprocessing incoming text
def preprocessing_text(text):
    text= text.lower()
    text= text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)


def train_model():
    print("Training model...")
    df=pd.read_csv('./spam_ham_dataset.csv')
    df['text']= df['text'].apply(lambda x: x.replace('\r\n',' '))

    corpus = df['text'].apply(preprocessing_text).values

    # convert text to feature vectors
    X= vectorizer.fit_transform(corpus).toarray()
    Y = df['label_num']

    # split the dataset into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # training the model
    clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(X_train, Y_train)

    # model evaluation
    print(f"Model trained with accuracy: {clf.score(X_test, Y_test):.2f}")

    # Save model and vectorizer
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)


# load the model and vectorizer from disk
def load_model_and_vectorizer():
    global clf, vectorizer
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)


if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    train_model()
load_model_and_vectorizer()

# spam prediction endpoint /predict-spam
@app.route('/predict-spam', methods=['POST'])
def predict_spam():
    if request.method == 'POST':
        data = request.get_json()
        comment = data.get('comment', '')

        preprocessed_comment= preprocessing_text(comment)

        # transform using trained vectorizer
        vect = vectorizer.transform([preprocessed_comment]).toarray()
        prediction = clf.predict(vect)
        
        return jsonify({
            'prediction': int(prediction[0]),
            'processed_text': preprocessed_comment  # For debugging
        })


if __name__ == '__main__':
    app.run(debug=True)