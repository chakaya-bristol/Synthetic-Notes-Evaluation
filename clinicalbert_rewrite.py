import os
import re
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)

os.environ["TRANSFORMERS_ALLOW_UNSAFE_DESERIALIZATION"] = "1"

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
PROC = Path("/projects/b5bg/processed")
OUT = Path("/projects/b5bg/results")
OUT.mkdir(parents=True, exist_ok=True)

N_REAL = 1000
MAX_LEN = 512
SEED = 42

# Same structure-stripping as the XGBoost pipeline (keep the two in sync) ──
def strip_structure(text):
    text = str(text)
    text = text.replace("<PHI>", " ")
    header_pattern = (
        r'(?im)^\s*(Name|Unit No|Admission Date|Discharge Date|Date of Birth|Sex|'
        r'Service|Attending|Allergies|Followup Instructions|Discharge Disposition):.*$'
    )
    text = re.sub(header_pattern, ' ', text)
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', ' ', text)
    return text

# Load real notes
with open(PROC / "test.txt", "r", encoding="utf-8") as f:
    real_texts = f.read().split("\n<|endoftext|>\n")
real_texts = [t.strip() for t in real_texts if t.strip()]
print(f"Loaded {len(real_texts)} real notes")

GENERATORS = {
    "qwen":       PROC / "synthetic_notes_qwen_rewrite.csv",
    "llama":      PROC / "synthetic_notes_llama_rewrite.csv",
}

def build_splits(synthetic_csv):
    """Same construction and split as the XGBoost pipeline, with structure stripped."""
    syn_df = pd.read_csv(synthetic_csv)
    random.seed(SEED)
    sampled_real = [strip_structure(t) for t in random.sample(real_texts, N_REAL)]
    syn_texts = [strip_structure(t) for t in syn_df["generated_text"].astype(str).tolist()]

    texts = sampled_real + syn_texts
    labels = [0] * N_REAL + [1] * len(syn_texts)

    tr_x, te_x, tr_y, te_y = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    return tr_x, te_x, tr_y, te_y

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": (preds == labels).mean(),
        "auc": roc_auc_score(labels, probs),
    }

def run(name, csv_path):
    print(f"\n{'='*60}\nClinicalBERT: Real vs {name}\n{'='*60}")
    tr_x, te_x, tr_y, te_y = build_splits(csv_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    def tok(batch):
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN
        )

    train_ds = Dataset.from_dict({"text": tr_x, "label": tr_y}).map(tok, batched=True)
    test_ds  = Dataset.from_dict({"text": te_x, "label": te_y}).map(tok, batched=True)

    args = TrainingArguments(
        output_dir=str(OUT / f"clinicalbert_rewrite_{name}"),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        save_total_limit=1,
        logging_steps=50,
        seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    save_dir = str(OUT / f"clinicalbert_rewrite_{name}")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved model to {save_dir}", flush=True)

    preds_out = trainer.predict(test_ds)
    probs = torch.softmax(torch.tensor(preds_out.predictions), dim=-1)[:, 1].numpy()
    preds = preds_out.predictions.argmax(axis=-1)

    print(classification_report(te_y, preds, target_names=["Real", "Synthetic"]))
    auc = roc_auc_score(te_y, probs)
    print(f"AUC-ROC: {auc:.4f}")

    pd.DataFrame({
        "text": te_x,
        "true_label": te_y,
        "pred_label": preds,
        "prob_synthetic": probs,
    }).to_csv(OUT / f"clinicalbert_rewrite_preds_{name}.csv", index=False)

    return {"generator": name, "auc": auc,
            "accuracy": float((preds == np.array(te_y)).mean())}

results = [run(name, path) for name, path in GENERATORS.items()]
pd.DataFrame(results).to_csv(OUT / "clinicalbert_rewrite_summary.csv", index=False)
print("\nSummary:")
print(pd.DataFrame(results))