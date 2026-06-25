import random
import pandas as pd
from transformers import pipeline
from pathlib import Path

MODEL_PATH = "/projects/b5bg/Models/distilgpt2-discharge-final"
TEST_PATH = "/projects/b5bg/processed/test.txt"
OUTPUT_PATH = "/projects/b5bg/processed/synthetic_notes_distilgpt2.csv"

# Load test notes
with open(TEST_PATH, "r", encoding="utf-8") as f:
    test_texts = f.read().split("\n<|endoftext|>\n")

test_texts = [t.strip() for t in test_texts if t.strip()]
print(f"Loaded {len(test_texts)} test notes")

# Extract meaningful prompt from each note
def get_prompt(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    skip_patterns = ["Name:", "Admission Date:", "Date of Birth:", "Unit No:"]

    for i, line in enumerate(lines):
        if any(line.startswith(p) for p in skip_patterns):
            continue
        # Return this line plus next line for more context
        if i + 1 < len(lines):
            return line + "\n" + lines[i + 1]
        return line

    return "History of Present Illness:"

# Sample 200 test notes
random.seed(42)
sampled = random.sample(test_texts, 200)
prompts = [get_prompt(t) for t in sampled]

# Load model
print("Loading model...")
generator = pipeline(
    "text-generation",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    clean_up_tokenization_spaces=False,
    device=0  # use GPU
)

# Generate synthetic notes
results = []
for i, prompt in enumerate(prompts):
    try:
        out = generator(
            prompt,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
            top_p=0.85,
            repetition_penalty=1.25,
            no_repeat_ngram_size=4,
            num_return_sequences=1
        )
        results.append({
            "prompt": prompt,
            "generated_text": out[0]["generated_text"]
        })
    except Exception as e:
        print(f"Error generating note {i}: {e}")
        results.append({
            "prompt": prompt,
            "generated_text": ""
        })

    if i % 20 == 0:
        print(f"Generated {i}/{len(prompts)} notes")

# Save
output_df = pd.DataFrame(results)
output_df = output_df[output_df["generated_text"].str.strip() != ""]
output_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {len(output_df)} synthetic notes to {OUTPUT_PATH}")