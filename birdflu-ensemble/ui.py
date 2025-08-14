"""Gradio Web UI for LLM Fine-tuning with LoRA."""

import gradio as gr
import json
import yaml
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import pandas as pd
from datasets import Dataset

# Import our training modules
from voters.llm.sft_lora import EnhancedLoRASFTTrainer
from scripts.infer_model import load_model, format_prompt, generate_response


class LLMFineTuningUI:
    """Web UI for LLM fine-tuning operations."""

    def __init__(self):
        self.config_path = "configs/llm_lora.yaml"
        self.load_config()

    def load_config(self):
        """Load configuration file."""
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            # Default config if file doesn't exist
            self.config = {
                "selected_model": "glm-4.5-air",
                "model_options": {
                    "glm-4.5-air": {"model_id": "ZHIPU-AI/glm-4-9b-chat"},
                    "qwen2.5-7b": {"model_id": "Qwen/Qwen2.5-7B"},
                    "mistral-7b": {"model_id": "mistralai/Mistral-7B-v0.1"},
                    "llama-3-8b": {"model_id": "meta-llama/Meta-Llama-3-8B"},
                },
                "lora": {"quantization": {"enabled": False}},
            }

    def save_config(self):
        """Save current configuration."""
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def update_training_config(
        self,
        model_name: str,
        lora_method: str,
        lora_rank: int,
        lora_alpha: int,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        quantization: str,
        scheduler_type: str,
    ) -> str:
        """Update training configuration based on UI inputs."""
        try:
            # Update model selection
            self.config["selected_model"] = (
                model_name.lower().replace("-", "_").replace(".", "_").replace(" ", "_")
            )

            # Update LoRA settings
            self.config["lora"]["method"] = lora_method.lower()
            self.config["lora"]["r"] = lora_rank
            self.config["lora"]["lora_alpha"] = lora_alpha

            # Update training settings
            self.config["training"]["learning_rate"] = learning_rate
            self.config["training"]["num_epochs"] = epochs
            self.config["training"]["batch_size"] = batch_size
            self.config["training"]["scheduler"]["type"] = scheduler_type.lower()

            # Update quantization
            if quantization == "None":
                self.config["lora"]["quantization"]["enabled"] = False
            else:
                self.config["lora"]["quantization"]["enabled"] = True
                self.config["lora"]["quantization"]["bits"] = int(quantization[0])

            self.save_config()
            return "✅ Configuration updated successfully!"

        except Exception as e:
            return f"❌ Error updating configuration: {str(e)}"

    def prepare_dataset(self, dataset_file) -> str:
        """Prepare dataset from uploaded file."""
        try:
            if dataset_file is None:
                return "❌ No dataset file uploaded"

            # Create data directory if it doesn't exist
            data_dir = Path("data/raw")
            data_dir.mkdir(parents=True, exist_ok=True)

            # Save uploaded file
            file_path = data_dir / "uploaded_dataset.json"

            # Read uploaded file content
            if hasattr(dataset_file, "name"):
                # File object
                with open(dataset_file.name, "r") as f:
                    data = json.load(f)
            else:
                # Direct content
                data = json.loads(dataset_file)

            # Validate dataset format
            if not isinstance(data, list):
                return "❌ Dataset must be a JSON list of examples"

            if len(data) == 0:
                return "❌ Dataset is empty"

            # Check required fields
            required_fields = ["text", "output"]
            first_example = data[0]
            missing_fields = [field for field in required_fields if field not in first_example]
            if missing_fields:
                return f"❌ Missing required fields: {missing_fields}"

            # Save processed dataset
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)

            return f"✅ Dataset prepared successfully! {len(data)} examples saved to {file_path}"

        except Exception as e:
            return f"❌ Error preparing dataset: {str(e)}"

    def start_training(self, progress=gr.Progress()) -> str:
        """Start the training process."""
        try:
            progress(0, desc="Initializing training...")

            # Check if dataset exists
            dataset_path = Path("data/raw/uploaded_dataset.json")
            if not dataset_path.exists():
                return "❌ No dataset found. Please upload a dataset first."

            # Initialize trainer
            trainer = EnhancedLoRASFTTrainer(self.config_path)

            progress(0.1, desc="Loading dataset...")

            # Load dataset
            with open(dataset_path, "r") as f:
                data = json.load(f)
            dataset = Dataset.from_list(data)

            progress(0.2, desc="Starting training...")

            # Start training (this will take a while)
            trainer.train(dataset)

            progress(1.0, desc="Training completed!")

            return "✅ Training completed successfully! Model saved to artifacts/models/llm_lora/"

        except Exception as e:
            return f"❌ Training failed: {str(e)}"

    def run_inference(
        self, text: str, model_path: str, confidence_threshold: float
    ) -> Tuple[str, str, str]:
        """Run inference on input text."""
        try:
            if not text.strip():
                return "❌ Please enter text to classify", "", ""

            # Check if model exists
            if not os.path.exists(model_path):
                return f"❌ Model not found at {model_path}", "", ""

            # Load model and run inference
            config = self.config
            model, tokenizer = load_model(config, model_path)

            # Format prompt and generate response
            prompt = format_prompt(text, {}, config)
            response = generate_response(model, tokenizer, prompt)

            # Try to parse JSON response
            try:
                result = json.loads(response)
                decision = result.get("decision", "Unknown")
                rationale = result.get("rationale", "No rationale provided")
                abstain = result.get("abstain", False)

                if abstain:
                    status = "🤖 Model abstained from prediction"
                else:
                    status = f"✅ Prediction: {decision}"

                return status, rationale, response

            except json.JSONDecodeError:
                return "⚠️ Model output (not valid JSON)", response, response

        except Exception as e:
            return f"❌ Inference failed: {str(e)}", "", ""

    def get_training_status(self) -> str:
        """Get current training status."""
        log_dir = Path("artifacts/models/llm_lora/logs")
        if log_dir.exists():
            # Check for recent log files
            log_files = list(log_dir.glob("**/events.out.tfevents.*"))
            if log_files:
                latest_log = max(log_files, key=os.path.getctime)
                mod_time = os.path.getmtime(latest_log)
                time_diff = time.time() - mod_time

                if time_diff < 300:  # Last 5 minutes
                    return "🟢 Training in progress..."
                else:
                    return "🟡 Training may have finished"
            else:
                return "⚪ No training logs found"
        else:
            return "⚪ No training started yet"

    def list_available_models(self) -> List[str]:
        """List available trained models."""
        models_dir = Path("artifacts/models")
        if not models_dir.exists():
            return []

        model_paths = []
        for path in models_dir.rglob("pytorch_model.bin"):
            model_paths.append(str(path.parent))

        return model_paths


def create_ui():
    """Create and configure the Gradio interface."""
    ui = LLMFineTuningUI()

    with gr.Blocks(title="LLM Fine-tuning with LoRA", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 LLM Fine-tuning with LoRA")
        gr.Markdown(
            "Fine-tune large language models using LoRA, QLoRA, and DoRA with support for multiple model architectures."
        )

        with gr.Tabs():
            # Training Tab
            with gr.Tab("🏋️ Training"):
                gr.Markdown("## Configure Training Parameters")

                with gr.Row():
                    with gr.Column():
                        model_dropdown = gr.Dropdown(
                            choices=["GLM-4.5-Air", "Qwen2.5-7B", "Mistral-7B", "Llama-3-8B"],
                            value="GLM-4.5-Air",
                            label="Base Model",
                        )
                        lora_method = gr.Radio(
                            choices=["LoRA", "DoRA"], value="LoRA", label="Training Method"
                        )
                        quantization = gr.Dropdown(
                            choices=["None", "4-bit", "8-bit"],
                            value="None",
                            label="Quantization (QLoRA)",
                        )

                    with gr.Column():
                        lora_rank = gr.Slider(
                            minimum=8, maximum=64, step=8, value=16, label="LoRA Rank"
                        )
                        lora_alpha = gr.Slider(
                            minimum=16, maximum=64, step=8, value=32, label="LoRA Alpha"
                        )
                        learning_rate = gr.Number(value=2e-4, label="Learning Rate")

                    with gr.Column():
                        epochs = gr.Slider(minimum=1, maximum=10, step=1, value=3, label="Epochs")
                        batch_size = gr.Slider(
                            minimum=1, maximum=16, step=1, value=4, label="Batch Size"
                        )
                        scheduler = gr.Dropdown(
                            choices=["cosine", "linear", "none"],
                            value="cosine",
                            label="Learning Rate Scheduler",
                        )

                with gr.Row():
                    update_config_btn = gr.Button("📝 Update Configuration", variant="secondary")
                    config_status = gr.Textbox(label="Configuration Status", interactive=False)

                gr.Markdown("## Upload Training Dataset")
                gr.Markdown(
                    'Upload a JSON file with examples in format: `[{"text": "input text", "output": "expected JSON output"}, ...]`'
                )

                with gr.Row():
                    dataset_file = gr.File(label="Dataset JSON File", file_types=[".json"])
                    dataset_status = gr.Textbox(label="Dataset Status", interactive=False)

                with gr.Row():
                    prepare_dataset_btn = gr.Button("📁 Prepare Dataset", variant="secondary")
                    start_training_btn = gr.Button("🚀 Start Training", variant="primary")

                training_output = gr.Textbox(label="Training Output", interactive=False, lines=3)

                # Real-time status
                with gr.Row():
                    status_btn = gr.Button("🔄 Check Status", variant="secondary")
                    training_status = gr.Textbox(label="Training Status", interactive=False)

            # Inference Tab
            with gr.Tab("🔮 Inference"):
                gr.Markdown("## Test Your Fine-tuned Model")

                with gr.Row():
                    with gr.Column():
                        input_text = gr.Textbox(
                            label="Input Text", placeholder="Enter text to classify...", lines=5
                        )
                        model_path = gr.Textbox(
                            label="Model Path",
                            value="artifacts/models/llm_lora/final",
                            placeholder="Path to your trained model",
                        )
                        confidence_threshold = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.8,
                            label="Confidence Threshold",
                        )

                        infer_btn = gr.Button("🔮 Run Inference", variant="primary")

                    with gr.Column():
                        prediction_result = gr.Textbox(label="Prediction Result", interactive=False)
                        rationale = gr.Textbox(label="Model Rationale", interactive=False, lines=3)
                        raw_output = gr.Textbox(
                            label="Raw Model Output", interactive=False, lines=5
                        )

            # Model Management Tab
            with gr.Tab("📁 Model Management"):
                gr.Markdown("## Available Models")

                with gr.Row():
                    list_models_btn = gr.Button("📋 List Models", variant="secondary")
                    available_models = gr.Textbox(
                        label="Available Models", interactive=False, lines=10
                    )

                gr.Markdown("## Model Information")
                model_info = gr.JSON(label="Model Configuration")

        # Event handlers
        update_config_btn.click(
            ui.update_training_config,
            inputs=[
                model_dropdown,
                lora_method,
                lora_rank,
                lora_alpha,
                learning_rate,
                epochs,
                batch_size,
                quantization,
                scheduler,
            ],
            outputs=config_status,
        )

        prepare_dataset_btn.click(ui.prepare_dataset, inputs=[dataset_file], outputs=dataset_status)

        start_training_btn.click(ui.start_training, outputs=training_output)

        status_btn.click(ui.get_training_status, outputs=training_status)

        infer_btn.click(
            ui.run_inference,
            inputs=[input_text, model_path, confidence_threshold],
            outputs=[prediction_result, rationale, raw_output],
        )

        list_models_btn.click(
            lambda: "\n".join(ui.list_available_models()), outputs=available_models
        )

    return demo


def main():
    """Launch the Gradio interface."""
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True for public sharing
        show_error=True,
    )


if __name__ == "__main__":
    main()
