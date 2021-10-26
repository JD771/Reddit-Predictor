import os
import joblib
import numpy as np
import praw
import tensorflow as tf
import json
from flask import Flask, jsonify, request, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)

saved_model = tf.keras.models.load_model('./best_model.h5')
labelencoder = joblib.load('./label_encoder.joblib')
tokenizer = joblib.load('./tokenizer.joblib')


def praw_extractor(url):
    # Import the environment variables for reddit authentication
    secret = os.environ['my_client_secret']
    reid = os.environ['my_client_id']
    # url = 'https://www.reddit.com/r/Economics/...'
    reddit = praw.Reddit(client_id=reid, client_secret=secret,
                         user_agent='Jahnavi WebScrapping')
    submission = reddit.submission(url=url)
    new_post = ''

    # Applying the top comments to new_post
    submission.comments.replace_more(limit=0)
    for comment in submission.comments:
        new_post = new_post + comment.body + ' '
    result = submission.link_flair_text
    return (new_post)


def predictor(post):
    seq = tokenizer.texts_to_sequences([post])
    padded = pad_sequences(seq, maxlen=350, padding='post', truncating='post')
    pred = saved_model.predict(padded)
    prediction = labelencoder.inverse_transform([np.argmax(pred)])
    return prediction


@app.route('/', methods=['GET'])
def home():
    return render_template('base.html')


@app.route('/predict', methods=['POST'])
def predict_items():
    data = request.get_json(force=True)
    new_post = praw_extractor(data['url'])
    prediction = predictor(new_post)
    return prediction[0]


@app.route('/automated_testing', methods=['POST'])
def automate():
    if request.method == 'POST':
        fileGiven = request.files['upload_file']
        urls = fileGiven.read().decode().split('\n')
        ans = {}
        for link in urls:
            link = link.replace('\r', '')
            post = praw_extractor(link)
            predicted_flair = predictor(post)[0]
            ans[link] = predicted_flair
        return json.dumps(ans)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)