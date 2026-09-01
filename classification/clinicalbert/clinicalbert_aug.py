import os
import re
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

MAX_LEN = 512
SEED = 42

# extended stripper — MUST match the XGBoost pipeline exactly
def strip_structure(text):
    text = str(text)
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'<?PHI\w*', ' ', text)
    header_pattern = (
        r'(?im)^\s*(Name|Unit No|Unit Number|Admission Date|Discharge Date|Date of Birth|'
        r'Sex|Gender|Service|Attending|Attending Physician|Allergies|Allergy|'
        r'Followup Instructions|Discharge Disposition):.*$'
    )
    text = re.sub(header_pattern, ' ', text)
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', ' ', text)
    return text
    
# Load the pre-built augmented dataset
df = pd.read_csv(PROC / "augmented_dataset.csv").dropna(subset=["text"])
print(f"Loaded {len(df)} notes")
print(f"Binary balance: {df['label'].value_counts().to_dict()}")

texts = [strip_structure(t) for t in df["text"].astype(str).tolist()]
labels = df["label"].tolist()

tr_x, te_x, tr_y, te_y = train_test_split(
    texts, labels, test_size=0.2, random_state=SEED, stratify=labels
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": (preds == labels).mean(),
        "auc": roc_auc_score(labels, probs),
    }

print(f"\n{'='*60}\nClinicalBERT: Real vs Synthetic (pooled)\n{'='*60}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def tok(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

train_ds = Dataset.from_dict({"text": tr_x, "label": tr_y}).map(tok, batched=True)
test_ds  = Dataset.from_dict({"text": te_x, "label": te_y}).map(tok, batched=True)

args = TrainingArguments(
    output_dir=str(OUT / "clinicalbert_augmented"),
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
}).to_csv(OUT / "clinicalbert_preds_augmented.csv", index=False)
print("Saved predictions to clinicalbert_preds_augmented.csv")