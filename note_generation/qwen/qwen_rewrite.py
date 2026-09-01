import os
import random
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

MODEL_PATH = "/projects/b5bg/models/qwen2.5-14b-instruct"
TRAIN_PATH = "/projects/b5bg/processed/train.txt"
OUTPUT_PATH = "/projects/b5bg/processed/synthetic_notes_qwen_rewrite.csv"

N_REWRITE = 1000

# Load real notes from the TRAIN split (never test — keeps test held out)
print("Loading train notes...")
with open(TRAIN_PATH, "r", encoding="utf-8") as f:
    train_texts = f.read().split("\n<|endoftext|>\n")
train_texts = [t.strip() for t in train_texts if t.strip()]
print(f"Loaded {len(train_texts)} train notes")

# Sample the real notes to rewrite
random.seed(42)
to_rewrite = random.sample(train_texts, N_REWRITE)

# Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype="auto", device_map="auto"
)
print("Model loaded")

def build_messages(real_note):
    return [
        {
            "role": "system",
            "content": (
                "You are a clinical documentation assistant. You will be given a real hospital "
                "discharge summary. Rewrite it in different words while preserving ALL clinical "
                "content exactly: the same diagnoses, findings, medications, doses, lab values, "
                "and the same overall structure and section order. Change only the phrasing and "
                "sentence construction — do not add, remove, or alter any clinical fact. "
                "Keep <PHI> placeholders wherever they appear. Use plain text only, no markdown."
            )
        },
        {
            "role": "user",
            "content": (
                "Rewrite the following discharge summary, preserving all clinical content and "
                "keeping <PHI> placeholders:\n\n" + real_note[:4000]
            )
        }
    ]

results = []
for i, real_note in enumerate(to_rewrite):
    messages = build_messages(real_note)
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.85,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )
        rewritten = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        results.append({"original_text": real_note, "generated_text": rewritten})
    except Exception as e:
        print(f"Error at note {i}: {e}")
        results.append({"original_text": real_note, "generated_text": ""})

    if i % 100 == 0 and i > 0:
        pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
        print(f"Rewritten {i}/{N_REWRITE} notes — checkpoint saved", flush=True)

output_df = pd.DataFrame(results)
output_df = output_df[output_df["generated_text"].str.strip() != ""]
output_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {len(output_df)} rewritten notes to {OUTPUT_PATH}")