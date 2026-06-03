# -*- coding: utf-8 -*-
"""Download ruBert-base model and tokenizer to a local directory."""

from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_NAME = "DeepPavlov/rubert-base-cased"
SAVE_DIR = "models/rubert-base-cased"

if __name__ == "__main__":
    print(f"Downloading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Tokenizer saved to {SAVE_DIR}")

    print(f"Downloading model: {MODEL_NAME}")
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    model.save_pretrained(SAVE_DIR)
    print(f"Model saved to {SAVE_DIR}")

    print("Done!")
