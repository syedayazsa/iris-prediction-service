"""
A Gradio-based front-end for interactive Iris model predictions.
"""

import gradio as gr
from typing import Tuple
import requests
import json


class GradioIrisDemo:
    """
    Encapsulates a Gradio interface for the Iris model.
    """

    def __init__(self, api_url: str = "http://localhost:8000"):
        """
        Initialize the GradioIrisDemo with the API URL.

        Args:
            api_url (str): The URL of the API for predictions.
        """
        self.api_url = api_url

    def predict_single_sample(self, sepal_length: float, sepal_width: float,
                              petal_length: float, petal_width: float) -> str:
        """
        Gradio callback to predict the species given a single flower's measurements.
        Returns the class label.

        Args:
            sepal_length (float): Length of the sepal in cm.
            sepal_width (float): Width of the sepal in cm.
            petal_length (float): Length of the petal in cm.
            petal_width (float): Width of the petal in cm.

        Returns:
            str: Predicted class label of the Iris species.
        """
        inputs = [[sepal_length, sepal_width, petal_length, petal_width]]
        response = requests.post(f"{self.api_url}/predict", json={"input": inputs})
        if response.status_code == 200:
            return response.json()["prediction"][0]
        return f"Error: {response.text}"

    def predict_with_confidence(self, sepal_length: float, sepal_width: float,
                                petal_length: float, petal_width: float) -> Tuple[str, str]:
        """
        Gradio callback returning predicted species AND a probability distribution for transparency.

        Args:
            sepal_length (float): Length of the sepal in cm.
            sepal_width (float): Width of the sepal in cm.
            petal_length (float): Length of the petal in cm.
            petal_width (float): Width of the petal in cm.

        Returns:
            Tuple[str, str]: Predicted class label and formatted probabilities.
        """
        inputs = [[sepal_length, sepal_width, petal_length, petal_width]]
        response = requests.post(f"{self.api_url}/predict-proba", json={"input": inputs})
        
        if response.status_code != 200:
            return "Error", f"API request failed: {response.text}"
            
        result = response.json()
        predicted = result["prediction"][0]
        probs = result["probabilities"][0]
        formatted_probs = f"Probabilities => setosa: {probs[0]:.2f}, versicolor: {probs[1]:.2f}, virginica: {probs[2]:.2f}"
        return predicted, formatted_probs

    def launch(self):
        """
        Create and launch the Gradio interface.
        """
        with gr.Blocks() as demo:
            gr.Markdown("## AyazIris Species Prediction Demo")
            gr.Markdown(
                "Adjust the sliders below to set the iris flower measurements, then click **Predict**!"
            )

            with gr.Row():
                inp_sepal_length = gr.Slider(0.0, 10.0, value=5.1, step=0.1, label="Sepal Length (cm)")
                inp_sepal_width = gr.Slider(0.0, 10.0, value=3.5, step=0.1, label="Sepal Width (cm)")

            with gr.Row():
                inp_petal_length = gr.Slider(0.0, 10.0, value=1.4, step=0.1, label="Petal Length (cm)")
                inp_petal_width = gr.Slider(0.0, 10.0, value=0.2, step=0.1, label="Petal Width (cm)")

            # Option 1: Single predicted label
            predict_button_label = gr.Button("Predict Species")
            output_label = gr.Textbox(label="Predicted Iris Species")

            predict_button_label.click(
                fn=self.predict_single_sample,
                inputs=[inp_sepal_length, inp_sepal_width, inp_petal_length, inp_petal_width],
                outputs=output_label
            )

            # Option 2: Predicted label + probability distribution
            gr.Markdown("### See Confidence Scores")
            predict_button_conf = gr.Button("Predict with Confidence")
            output_label_conf = gr.Textbox(label="Predicted Species")
            output_probs = gr.Textbox(label="Class Probabilities")

            predict_button_conf.click(
                fn=self.predict_with_confidence,
                inputs=[inp_sepal_length, inp_sepal_width, inp_petal_length, inp_petal_width],
                outputs=[output_label_conf, output_probs]
            )

        # Launch Gradio app
        demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    GradioIrisDemo().launch()