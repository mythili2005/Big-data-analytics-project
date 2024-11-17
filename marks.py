import joblib

# Load the trained models
lr_model = joblib.load(r'C:\Users\Admin\Documents\bda\models\lr_model.pkl')  # Logistic Regression model
tokenizer = joblib.load(r'C:\Users\Admin\Documents\bda\models\tokenizer_inputCol.pkl')  # Tokenizer
hashingTF = joblib.load(r'C:\Users\Admin\Documents\bda\models\hashingTF_numFeatures.pkl')  # HashingTF

def analyze_tweet(tweet):
    # Preprocess and vectorize the tweet using tokenizer and HashingTF
    tweet_transformed = tokenizer.transform([tweet])
    tweet_vectorized = hashingTF.transform(tweet_transformed)
    
    # Make prediction using the Logistic Regression model
    prediction = lr_model.predict(tweet_vectorized)
    
    # Interpret prediction (example: 1 = Positive, 0 = Negative)
    if prediction == 1:
        return "Positive Sentiment"
    else:
        return "Negative Sentiment"
