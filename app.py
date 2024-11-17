from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the pre-trained model
pipeline = joblib.load('logistic_regression_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sub', methods=['POST'])
def analyze_sentiment():
    tweet = request.form['tweet']  # Get tweet from the form input
    
    # Make prediction using the model
    prediction = pipeline.predict([tweet])
    
    # Return the result as a response
    if prediction[0] == 1:
        sentiment = "Positive Sentiment 😊"
    else:
        sentiment = "Negative Sentiment 😞"
    
    return render_template('index.html', sentiment=sentiment, tweet=tweet)

if __name__ == '__main__':
    app.run(debug=True)
