
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from flask import Flask, request, jsonify

nltk.download('vader_lexicon')

app = Flask(__name__)

@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.json
    text = data['text']

    # Sentiment analysis using NLTK's SentimentIntensityAnalyzer
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = sentiment_analyzer.polarity_scores(text)

    # Determine sentiment label based on compound score
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        sentiment_label = 'Positive'
    elif compound_score <= -0.05:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'

    result = {
        'text': text,
        'sentiment': sentiment_label,
        'sentiment_scores': sentiment_scores
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
