import numpy as np
from tensorflow.keras.models import load_model, Model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input

# Load the pre-trained model and tokenizer
loaded_model = load_model("model/sentiment_model.h5")
with open("model/tokenizer.pkl", "rb") as f:
    loaded_tokenizer = pickle.load(f)

def sentiment(text):
    test_sequence = loaded_tokenizer.texts_to_sequences([text])
    max_sequence_length = 200  
    padded_test_sequence = pad_sequences(test_sequence, maxlen=max_sequence_length)

    # Make predictions
    prediction = loaded_model.predict(padded_test_sequence)

    # Convert prediction to sentiment
    sentiment_label = np.argmax(prediction)
    sentiment_map = {0: "negative", 1: "positive", 2: "neutral"}
    predicted_sentiment = sentiment_map[sentiment_label]
    return predicted_sentiment
