from pathlib import Path
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt

from src.predictor import EmotionPredictor

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
LABELS = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]

predictor = EmotionPredictor(MODELS_DIR)
available_models = predictor.available_models()

def predict(text, model_name):
    if not text or not text.strip():
        raise gr.Error("Please enter a text message.")
    if not model_name:
        raise gr.Error("No exported model was found. Add your Kaggle model exports to the models/ folder.")

    result = predictor.predict(text.strip(), model_name)
    scores = pd.DataFrame({
        "Emotion": list(result["scores"].keys()),
        "Confidence": list(result["scores"].values())
    }).sort_values("Confidence", ascending=False)

    attention_fig = None
    if result.get("attention"):
        attention_fig, ax = plt.subplots(figsize=(max(6, len(result["attention"]) * 0.45), 2.8))
        values = [item["weight"] for item in result["attention"]]
        tokens = [item["token"] for item in result["attention"]]
        ax.imshow([values], aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_yticks([])
        ax.set_title("BiLSTM Attention Weights")
        ax.set_xlabel("Input tokens")
        attention_fig.tight_layout()

    return result["emotion"], scores, attention_fig

with gr.Blocks(title="Emotion Classification with Attention", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Emotion Classification with Attention
        Enter a sentence to classify it into one of six emotions.
        Select one of the trained models and compare its confidence scores.
        """
    )
    with gr.Row():
        model_choice = gr.Dropdown(
            choices=available_models,
            value=available_models[0] if available_models else None,
            label="Model",
            interactive=True
        )
    text_input = gr.Textbox(
        label="Input text",
        placeholder="Example: I am so happy that everything worked out!",
        lines=4
    )
    submit = gr.Button("Predict emotion", variant="primary")
    predicted_emotion = gr.Label(label="Predicted emotion")
    confidence_plot = gr.BarPlot(
        label="Confidence across the six emotions",
        x="Emotion",
        y="Confidence",
        vertical=False,
        y_lim=[0, 1]
    )
    attention_plot = gr.Plot(label="Attention visualization (BiLSTM + Attention only)")
    submit.click(
        fn=predict,
        inputs=[text_input, model_choice],
        outputs=[predicted_emotion, confidence_plot, attention_plot]
    )
    gr.Markdown(
        """
        **Note:** Attention visualization is available for the BiLSTM + Attention model.
        Other models return confidence scores without a token-level attention plot.
        """
    )

if __name__ == "__main__":
    demo.launch()
