import os
import random
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# Paths
MODEL_PATH = "/projects/b5bg/models/qwen2.5-14b-instruct"
TRAIN_PATH = "/projects/b5bg/processed/train.txt"
OUTPUT_PATH = "/projects/b5bg/processed/synthetic_notes_qwen.csv"

N_GENERATE = 1000
N_EXAMPLES = 3  # number of few-shot examples per prompt

# Load train notes
print("Loading train notes...")
with open(TRAIN_PATH, "r", encoding="utf-8") as f:
    train_texts = f.read().split("\n<|endoftext|>\n")
train_texts = [t.strip() for t in train_texts if t.strip()]
print(f"Loaded {len(train_texts)} train notes")

# Sample 200 notes to use as few-shot example pool
random.seed(42)
example_pool = random.sample(train_texts, 200)

# Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"
)
print("Model loaded")

def build_prompt(examples):
    """Build a few-shot prompt using example discharge summaries."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a clinical documentation assistant specialising in hospital discharge summaries. "
                "You will be shown real examples of discharge summaries and must generate a new one "
                "that follows the same structure, style, clinical terminology, and level of detail. "
                "Use <PHI> as a placeholder wherever patient-identifiable information would appear."
                "Do not use markdown formatting such as ** or ### — use plain text only."
            )
        }
    ]

    # Add examples as user/assistant turns
    for i, ex in enumerate(examples):
        messages.append({
            "role": "user",
            "content": f"Here is example discharge summary {i+1}:\n\n{ex[:1500]}"
        })
        messages.append({
            "role": "assistant",
            "content": "Understood. I have reviewed this discharge summary."
        })

    # Final instruction
    messages.append({
        "role": "user",
        "content": (
            "Now generate a completely new discharge summary in the same format and style as the examples above. "
            "It should be realistic and clinically plausible, with appropriate sections including "
            "Chief Complaint, History of Present Illness, Past Medical History, Physical Exam, "
            "Pertinent Results, Brief Hospital Course, Discharge Medications, and Discharge Instructions."
            "Do not use markdown formatting such as ** or ### — use plain text only."
        )
    })

    return messages

# Generate notes
results = []
for i in range(N_GENERATE):
    # Pick 3 random examples from the pool for each generation
    examples = random.sample(example_pool, N_EXAMPLES)
    messages = build_prompt(examples)

    try:
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=800,
            do_sample=True,
            temperature=0.7,
            top_p=0.85,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode only the newly generated tokens
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        results.append({
            "generated_text": generated
        })

    except Exception as e:
        print(f"Error at note {i}: {e}")
        results.append({"generated_text": ""})

    if i % 50 == 0:
        print(f"Generated {i}/{N_GENERATE} notes")

# Save
output_df = pd.DataFrame(results)
output_df = output_df[output_df["generated_text"].str.strip() != ""]
output_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {len(output_df)} synthetic notes to {OUTPUT_PATH}")
