import os, re, numpy as np, pandas as pd, torch, shap
os.environ["TRANSFORMERS_ALLOW_UNSAFE_DESERIALIZATION"] = "1"
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from collections import defaultdict

TAGS = ["rewrite_qwen", "rewrite_llama", "freegen_qwen", "freegen_llama", "augmented"]

def ckpt_for(tag):
    p = f"/projects/b5bg/results/clinicalbert_{tag}"
    return p + "/checkpoint-800" if tag == "augmented" else p

def preds_for(tag):
    name = {"rewrite_qwen": "qwen", "rewrite_llama": "llama"}.get(tag, tag)
    return f"/projects/b5bg/results/clinicalbert_preds_{name}.csv"

# strip_structure — MUST be byte-identical to the training script
def strip_structure(text):
    text = str(text)
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'<?PHI\w*', ' ', text)
    text = re.sub(
        r'(?im)^\s*(Name|Unit No|Unit Number|Admission Date|Discharge Date|Date of Birth|'
        r'Sex|Gender|Service|Attending|Attending Physician|Allergies|Allergy|'
        r'Followup Instructions|Discharge Disposition):.*$', ' ', text)
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', ' ', text)
    return text

for TAG in TAGS:
    CKPT  = ckpt_for(TAG)
    PREDS = preds_for(TAG)
    print(f"\n{'='*60}\n{TAG}\n{'='*60}", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(CKPT)
        model = AutoModelForSequenceClassification.from_pretrained(CKPT).eval().cuda()
        clf_pipe = pipeline("text-classification", model=model, tokenizer=tok,
                            device=0, top_k=None, truncation=True, max_length=512)
        explainer = shap.Explainer(clf_pipe)

        pr = pd.read_csv(PREDS)
        pr["input"] = pr["text"].map(strip_structure)
        sample = pr.sample(min(150, len(pr)), random_state=42)
        sv = explainer(sample["input"].tolist())

        agg = defaultdict(list)
        for row in sv:
            vals = row.values[:, 1] if row.values.ndim > 1 else row.values
            for tokstr, v in zip(row.data, vals):
                t = tokstr.strip().lower()
                if t: agg[t].append(float(v))
        ranked = sorted(((np.mean(v), t, len(v)) for t, v in agg.items() if len(v) >= 5), reverse=True)

        print(f"--- {TAG}: toward SYNTHETIC ---", flush=True)
        for m, t, n in ranked[:25]: print(f"  {t:22s} {m:+.4f} (n={n})")
        print(f"--- {TAG}: toward REAL ---", flush=True)
        for m, t, n in ranked[-25:]: print(f"  {t:22s} {m:+.4f} (n={n})")

        pd.DataFrame(ranked, columns=["mean_shap","token","count"]).to_csv(
            f"/projects/b5bg/results/cbert_tokens_global_{TAG}.csv", index=False)
        print(f"saved cbert_tokens_global_{TAG}.csv", flush=True)

        del model, clf_pipe, explainer
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"!! {TAG} failed: {e}", flush=True)
        continue