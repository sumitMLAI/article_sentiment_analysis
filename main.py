from flask import Flask, request, jsonify, render_template
from component.summary_article import summarize_text
from component.sentiment_analysis import sentiment

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def get_summary():
    data = request.get_json()
    text = data['text']
    summary = summarize_text(text)
    return jsonify({'summary': summary})

@app.route('/sentiment', methods=['POST'])
def get_sentiment():
    data = request.get_json()
    text = data['text']
    sentiment_result = sentiment(text)
    return jsonify({'sentiment': sentiment_result})

if __name__ == '__main__':
    app.run(debug=True)
