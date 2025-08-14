"""Script to merge LoRA adapters into base model for deployment."""

import argparse
import json
import yaml
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load model configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def merge_lora_adapter(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
    config_path: str = "configs/llm_lora.yaml",
):
    """
    Merge LoRA adapter with base model.

    Args:
        base_model_path: Path to base model or HuggingFace model ID
        lora_adapter_path: Path to LoRA adapter
        output_path: Path to save merged model
        config_path: Path to configuration file
    """
    logger.info(f"Merging LoRA adapter from {lora_adapter_path}")
    logger.info(f"Base model: {base_model_path}")
    logger.info(f"Output path: {output_path}")

    # Load configuration
    config = load_config(config_path)

    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base model
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Load on CPU for merging
        trust_remote_code=True,
    )

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # Load LoRA adapter
    logger.info("Loading LoRA adapter...")
    model_with_adapter = PeftModel.from_pretrained(base_model, lora_adapter_path, device_map="cpu")

    # Merge adapter weights into base model
    logger.info("Merging adapter weights...")
    merged_model = model_with_adapter.merge_and_unload()

    # Save merged model
    logger.info("Saving merged model...")
    merged_model.save_pretrained(output_path, safe_serialization=True, max_shard_size="5GB")

    # Save tokenizer
    tokenizer.save_pretrained(output_path)

    # Save configuration info
    config_info = {
        "base_model": base_model_path,
        "lora_adapter": lora_adapter_path,
        "merged_at": str(output_path),
        "model_type": "merged_lora",
    }

    with open(output_dir / "merge_info.json", "w") as f:
        json.dump(config_info, f, indent=2)

    logger.info(f"✅ Merge completed successfully!")
    logger.info(f"Merged model saved to: {output_path}")

    return output_path


def verify_merged_model(model_path: str, test_text: str = "Hello, how are you?"):
    """
    Verify that the merged model works correctly.

    Args:
        model_path: Path to merged model
        test_text: Test text for generation
    """
    logger.info(f"Verifying merged model at {model_path}")

    try:
        # Load merged model
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Test generation
        inputs = tokenizer(test_text, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"✅ Model verification successful!")
        logger.info(f"Test input: {test_text}")
        logger.info(f"Generated output: {generated_text}")

        return True

    except Exception as e:
        logger.error(f"❌ Model verification failed: {e}")
        return False


def get_model_size(model_path: str) -> str:
    """Get the size of the merged model."""
    model_dir = Path(model_path)
    total_size = 0

    for file_path in model_dir.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size

    # Convert to GB
    size_gb = total_size / (1024**3)
    return f"{size_gb:.2f} GB"


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument(
        "--base-model", required=True, help="Base model path or HuggingFace model ID"
    )
    parser.add_argument("--lora-path", required=True, help="Path to LoRA adapter")
    parser.add_argument("--output-path", required=True, help="Output path for merged model")
    parser.add_argument("--config", default="configs/llm_lora.yaml", help="Configuration file path")
    parser.add_argument("--verify", action="store_true", help="Verify merged model after creation")
    parser.add_argument(
        "--test-text",
        default="Classify this text about bird flu research.",
        help="Test text for verification",
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.lora_path).exists():
        logger.error(f"LoRA adapter path does not exist: {args.lora_path}")
        return

    if not Path(args.config).exists():
        logger.error(f"Configuration file does not exist: {args.config}")
        return

    # Perform merge
    try:
        output_path = merge_lora_adapter(
            args.base_model, args.lora_path, args.output_path, args.config
        )

        # Get model size
        model_size = get_model_size(output_path)
        logger.info(f"Merged model size: {model_size}")

        # Verify if requested
        if args.verify:
            logger.info("Verifying merged model...")
            verify_merged_model(output_path, args.test_text)

        logger.info("🎉 Merge process completed successfully!")

    except Exception as e:
        logger.error(f"❌ Merge failed: {e}")
        raise


if __name__ == "__main__":
    main()
