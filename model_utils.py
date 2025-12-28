import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax

MODEL_NAME = "cardiffnlp/twitter-roberta-base-irony"


def load_model():
    """
    Loads the pre-trained sarcasm/irony detection model and tokenizer.
    """
    print("Loading model... (this may take a moment)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    print("Model loaded successfully!")
    return tokenizer, model


def predict_sarcasm(text, tokenizer, model):
    """
    Predicts whether a given text is sarcastic or literal.
    Returns label and confidence score.
    """
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    prediction_index = np.argmax(scores)
    label = "Sarcastic" if prediction_index == 1 else "Literal"
    confidence = scores[prediction_index]

    return label, confidence
