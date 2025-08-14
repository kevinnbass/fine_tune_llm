#!/usr/bin/env python3
"""
Simple inference script for fine-tuned GLM-4.5-Air model.
"""

import json
import torch
import yaml
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_config():
    """Load LoRA configuration."""
    config_path = Path("configs/llm_lora.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(config, model_path):
    """Load the fine-tuned model and tokenizer."""
    # Load base model and tokenizer
    model_id = config["lora"]["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    # Load LoRA adapter if path provided
    if model_path and Path(model_path).exists():
        model = PeftModel.from_pretrained(base_model, model_path)
        print(f"Loaded LoRA adapter from: {model_path}")
    else:
        model = base_model
        print("Using base model (no LoRA adapter)")

    return model, tokenizer


def format_prompt(text, metadata, config):
    """Format the input prompt according to configuration."""
    instruction_format = config["instruction_format"]

    system_prompt = instruction_format["system_prompt"].strip()
    input_template = instruction_format["input_template"]

    # Format the input
    formatted_input = input_template.format(
        text=text.strip(), metadata=json.dumps(metadata) if metadata else "{}"
    )

    # Create the full prompt
    prompt = f"{system_prompt}\n\n{formatted_input}\n\nResponse:"

    return prompt


def generate_response(model, tokenizer, prompt, max_length=512, use_structured=False, schema=None):
    """Generate response from the model with optional structured output."""

    if use_structured and schema:
        try:
            # Try to use outlines for structured generation
            import outlines

            # Create structured generator
            generator = outlines.generate.json(model, schema)
            response = generator(prompt)

            return json.dumps(response) if isinstance(response, dict) else str(response)

        except ImportError:
            print(
                "Warning: outlines not available for structured generation, falling back to regular generation"
            )
        except Exception as e:
            print(
                f"Warning: structured generation failed ({e}), falling back to regular generation"
            )

    # Regular generation
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the new tokens (response part)
    prompt_length = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
    generated_text = response[prompt_length:].strip()

    return generated_text


def parse_json_response(response_text):
    """Try to parse JSON response, return None if invalid."""
    try:
        # Look for JSON-like content
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1

        if start_idx >= 0 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def main():
    """Main inference function."""
    import argparse

    parser = argparse.ArgumentParser(description="Inference with fine-tuned models")
    parser.add_argument("--text", type=str, required=True, help="Text to classify")
    parser.add_argument("--model-path", type=str, help="Path to LoRA adapter")
    parser.add_argument("--metadata", type=str, help="JSON metadata string")
    parser.add_argument(
        "--structured", action="store_true", help="Use structured output generation"
    )
    parser.add_argument("--schema", type=str, help="JSON schema for structured output")
    parser.add_argument(
        "--config", type=str, default="configs/llm_lora.yaml", help="Config file path"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Parse metadata
    metadata = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            print("Warning: Invalid metadata JSON, using empty dict")

    # Get schema for structured output
    schema = None
    if args.structured:
        if args.schema:
            try:
                schema = json.loads(args.schema)
            except json.JSONDecodeError:
                print("Warning: Invalid schema JSON, using config default")
                schema = config.get("output", {}).get("schema")
        else:
            # Use schema from config
            schema_str = config.get("output", {}).get(
                "schema", '{"decision": "str", "rationale": "str", "abstain": "bool"}'
            )
            try:
                schema = json.loads(schema_str)
            except:
                schema = {"decision": "str", "rationale": "str", "abstain": "bool"}

    # Load model
    print("Loading model...")
    model, tokenizer = load_model(config, args.model_path)

    # Format prompt
    prompt = format_prompt(args.text, metadata, config)
    print(f"Prompt:\n{prompt}\n")

    # Generate response
    print("Generating response...")
    use_structured = args.structured and config.get("output", {}).get("structured", False)
    response = generate_response(
        model, tokenizer, prompt, use_structured=use_structured, schema=schema
    )
    print(f"Raw response: {response}")

    # Try to parse JSON
    parsed = parse_json_response(response)
    if parsed:
        print(f"Parsed JSON: {json.dumps(parsed, indent=2)}")

        # Validate against schema if provided
        if schema and use_structured:
            try:
                import jsonschema

                jsonschema.validate(parsed, schema)
                print("✅ Response validates against schema")
            except ImportError:
                print("jsonschema not available for validation")
            except jsonschema.ValidationError as e:
                print(f"❌ Schema validation failed: {e}")
    else:
        print("Could not parse response as JSON")


if __name__ == "__main__":
    main()
