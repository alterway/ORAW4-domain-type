from flask import Flask, jsonify, request
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words

app = Flask(__name__)

model_id = 0
TRUE_K = 5
TOKEN_PATTERN = '[A-Za-zÀ-ÖØ-öø-ÿ]{2,}'

@app.route('/train')
def train():
    if not request.args.get('dataset'):
        return '"dataset" query parameter is required', 400

    print('loading dataset')
    data = pd.read_csv(request.args.get('dataset'))
    stop_words = get_stop_words(request.args.get('lang', 'fr'))

    print('loading stop words')
    if request.args.get('stop_words'):
        stop_data = pd.read_csv(request.args.get('stop_words'))
        stop_words.extend(stop_data['stop_words'])
       
    vectorizer = TfidfVectorizer(stop_words=stop_words, token_pattern=TOKEN_PATTERN)
    x = vectorizer.fit_transform(data['text'])

    print('training model')
    model = KMeans(n_clusters=TRUE_K, init='k-means++', random_state=42)
    model.fit(x)

    print('saving model')
    pickle.dump([vectorizer, model], open('model_%d.pkl' % (model_id % 10), 'wb'))
    return jsonify(model_id)

@app.route('/predict')
def predict():
    if not request.args.get('model_id') or not request.args.get('text'):
        return '"model_id" query parameter is required', 400

    print('loading model')
    vectorizer, model = pickle.load(open('model_%s.pkl' % request.args.get('model_id'), 'rb'))

    print('predicting')
    return jsonify(model.predict(vectorizer.transform([request.args.get('text')])).tolist()[0])

@app.route('/healthcheck')
def healthcheck():
    return 'ok', 200

print('ready')