import os
import random
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# Paths
MODEL_PATH = "/projects/b5bg/models/llama-3.1-8b-instruct"
TEST_PATH = "/projects/b5bg/processed/test.txt"
OUTPUT_PATH = "/projects/b5bg/processed/synthetic_notes_llama.csv"

N_GENERATE = 1000
N_EXAMPLES = 3

# Load test notes
print("Loading test notes...")
with open(TEST_PATH, "r", encoding="utf-8") as f:
    test_texts = f.read().split("\n<|endoftext|>\n")
test_texts = [t.strip() for t in test_texts if t.strip()]
print(f"Loaded {len(test_texts)} test notes")

# Sample 200 notes as few-shot example pool
random.seed(42)
example_pool = random.sample(test_texts, 200)

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
    messages = [
        {
            "role": "system",
            "content": (
                "You are a clinical documentation assistant specialising in hospital discharge summaries. "
                "You will be shown real examples of discharge summaries and must generate a new one "
                "that follows the same structure, style, clinical terminology, and level of detail. "
                "Use <PHI> as a placeholder wherever patient-identifiable information would appear. "
                "Do not use markdown formatting such as ** or ### — use plain text only."
            )
        }
    ]

    for i, ex in enumerate(examples):
        messages.append({
            "role": "user",
            "content": f"Here is example discharge summary {i+1}:\n\n{ex[:1500]}"
        })
        messages.append({
            "role": "assistant",
            "content": "Understood. I have reviewed this discharge summary."
        })

    messages.append({
        "role": "user",
        "content": (
            "Now generate a completely new discharge summary in the same format and style as the examples above. "
            "It should be realistic and clinically plausible, with appropriate sections including "
            "Chief Complaint, History of Present Illness, Past Medical History, Physical Exam, "
            "Pertinent Results, Brief Hospital Course, Discharge Medications, and Discharge Instructions. "
            "Use plain text formatting only, no markdown."
        )
    })

    return messages

# Generate notes
results = []
for i in range(N_GENERATE):
    examples = random.sample(example_pool, N_EXAMPLES)
    messages = build_prompt(examples)

    try:
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

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        results.append({"generated_text": generated})

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