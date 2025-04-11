import base64
import pickle
import re

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

sentiment_bp = Blueprint('sentiment', __name__)

scaler_path = 'models/sentiment/scaler.pkl'
vectorizer_path = 'models/sentiment/countVectorizer.pkl'
model_lgbm_path = 'models/sentiment/model_lgbm.pkl'

STOPWORDS = set(stopwords.words("english"))
ps = PorterStemmer()


# Load the model and vectorizer
def single_prediction(text_input, cv, scaler, predictor):
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in STOPWORDS]
    review = " ".join(review)

    X_prediction = cv.transform([review]).toarray()
    X_prediction_scl = scaler.transform(X_prediction)

    y_prediction = predictor.predict(X_prediction_scl)[0]
    proba = predictor.predict_proba(X_prediction_scl)[0]

    sentiment_range = (proba[2] * 100 + proba[1] * 50 + proba[0] * 0)
    return {
        "y_prediction": y_prediction,
        "proba": proba.tolist(),
        "range": sentiment_range
    }


@sentiment_bp.route('/predict-sentiment', methods=['POST'])
def predict_sentiment():
    scaler = pickle.load(open(scaler_path, "rb"))
    cv = pickle.load(open(vectorizer_path, "rb"))
    predictor = pickle.load(open(model_lgbm_path, "rb"))
    try:
        if request.method == 'POST':
            text_input = request.json['text']

            sentiment_res= single_prediction(text_input, cv, scaler, predictor)
            print("Sentiment Result:", sentiment_res)
            return jsonify({
                "prediction": sentiment_res["y_prediction"], 
                "probability": sentiment_res["proba"],
                "range": round(sentiment_res["range"], 1),
                "text": text_input
                })
        
    except Exception as e:
        return jsonify({"error": str(e)})
