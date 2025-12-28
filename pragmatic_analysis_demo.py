# Sarcasm Detection Demo
# This script illustrates how to use a pre-trained AI model to detect sarcasm in text.
# It also generates the specific visualizations (Confusion Matrix & Accuracy) from the research poster.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, accuracy_score

# Setup plotting style for better visuals
sns.set_theme(style="whitegrid")

def plot_poster_visuals():
    """
    Generates the specific charts from the poster using the hardcoded research results.
    """
    print("\nGenerating Research Poster Visualizations...")
    
    # --- Chart 1: Confusion Matrix (Linguistic Failure) ---
    # Data from poster: TN=450, FP=85, FN=36, TP=120
    # Note: Structure is [[TN, FP], [FN, TP]]
    cm_data = np.array([[450, 85], [36, 120]])
    labels = ["Literal\n(Blue)", "Sarcastic\n(Orange)"]
    
    plt.figure(figsize=(8, 6))
    # Custom annotations with "True Negative", "False Positive" etc.
    annot_labels = [
        [f"450\n(True Negative)", f"85\n(False Positive)"],
        [f"36\n(False Negative\nMissed Sarcasm)", f"120\n(True Positive)"]
    ]
    
    # Create the heatmap
    ax = sns.heatmap(cm_data, annot=annot_labels, fmt='', cmap='Oranges', 
                     xticklabels=labels, yticklabels=labels, cbar=False,
                     annot_kws={"size": 12, "weight": "bold"})
    
    # Customizing colors manually to match the Blue/Orange theme roughly if possible,
    # but strictly 'Oranges' cmap is generic. Let's stick to a clean design.
    plt.xlabel('Predicted ML Label', fontweight='bold')
    plt.ylabel('Actual Linguistic Meaning', fontweight='bold')
    plt.title('Quantifying Linguistic Failure (Confusion Matrix)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    print(" - Displayed Confusion Matrix.")

    # --- Chart 2: Accuracy Comparison ---
    # Data from poster
    categories = ['Linguistic Limitations', 'High Accuracy']
    values = [78, 98]
    colors = ['#FFC000', '#0070C0'] # Gold/Orange and Blue
    
    plt.figure(figsize=(6, 5))
    bars = plt.bar(categories, values, color=colors, width=0.5)
    
    # Add labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height - 10,
                 f'{height}%',
                 ha='center', va='bottom', color='white', fontweight='bold', fontsize=14)
    
    plt.title('Performance Gap: Context Matters', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage')
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.show()
    print(" - Displayed Accuracy Comparison Chart.")


def main():
    # ---------------------------------------------------------
    # 1. Load the Pre-trained Model
    # ---------------------------------------------------------
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-irony"

    print("Loading model... (this may take a moment)")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have an internet connection and the 'transformers' library installed.")
        return

    # Helper function to run the model on text
    def predict_sarcasm_score(text):
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        prediction_index = np.argmax(scores)
        label = "Sarcastic" if prediction_index == 1 else "Literal"
        confidence = scores[prediction_index]
        return label, confidence

    # ---------------------------------------------------------
    # 2. Run Inference on Examples (Live Demo)
    # ---------------------------------------------------------
    examples = [
        {"text": "I love my job, it is very rewarding.", "truth": "Literal"},
        {"text": "Oh great, I missed the bus again.", "truth": "Sarcastic"},
        {"text": "The weather is just perfect for a picnic (in a storm).", "truth": "Sarcastic"}
    ]

    print("\n" + "="*80)
    print(f" LIVE DEMO: Running inference on {len(examples)} examples...")
    print(f"{'Text':<50} | {'Prediction':<10} | {'Confidence':<10}")
    print("="*80)

    for item in examples:
        text = item["text"]
        prediction, conf = predict_sarcasm_score(text)
        print(f"{text[:47]+'...':<50} | {prediction:<10} | {conf:.4f}")

    # ---------------------------------------------------------
    # 3. Show Poster Visualizations
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print(" RESEARCH DATA: Visualizing results from the study...")
    print("="*80)
    plot_poster_visuals()

if __name__ == "__main__":
    main()
