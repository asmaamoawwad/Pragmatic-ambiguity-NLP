from model_utils import load_model, predict_sarcasm
from visualization import plot_poster_visuals
from demo_examples import EXAMPLES


def run_live_demo(tokenizer, model):
    print("\n" + "=" * 80)
    print(f" LIVE DEMO: Running inference on {len(EXAMPLES)} examples...")
    print(f"{'Text':<50} | {'Prediction':<10} | {'Confidence':<10}")
    print("=" * 80)

    for item in EXAMPLES:
        text = item["text"]
        prediction, confidence = predict_sarcasm(text, tokenizer, model)
        print(f"{text[:47] + '...':<50} | {prediction:<10} | {confidence:.4f}")


def main():
    tokenizer, model = load_model()
    run_live_demo(tokenizer, model)

    print("\n" + "=" * 80)
    print(" RESEARCH DATA: Visualizing results from the study...")
    print("=" * 80)

    plot_poster_visuals()


if __name__ == "__main__":
    main()
