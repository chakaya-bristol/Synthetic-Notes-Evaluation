import os, re, numpy as np, pandas as pd, torch, shap
os.environ["TRANSFORMERS_ALLOW_UNSAFE_DESERIALIZATION"] = "1"
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

TAGS = ["rewrite_qwen", "rewrite_llama", "freegen_qwen", "freegen_llama", "augmented"]

def ckpt_for(tag):
    p = f"/projects/b5bg/results/clinicalbert_{tag}"
    return p + "/checkpoint-800" if tag == "augmented" else p

def preds_for(tag):
    name = {"rewrite_qwen": "qwen", "rewrite_llama": "llama"}.get(tag, tag)
    return f"/projects/b5bg/results/clinicalbert_preds_{name}.csv"

# strip_structure — MUST be byte-identical to the training script (clinicalbert_rewrite.py line 28)
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
    print(f"\n{'='*60}\n{TAG}\n  model: {CKPT}\n  preds: {PREDS}\n{'='*60}", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(CKPT)
        model = AutoModelForSequenceClassification.from_pretrained(CKPT).eval().cuda()
        clf_pipe = pipeline("text-classification", model=model, tokenizer=tok,
                            device=0, top_k=None, truncation=True, max_length=512)
        explainer = shap.Explainer(clf_pipe)

        pr = pd.read_csv(PREDS)
        pr["input"] = pr["text"].map(strip_structure)

        # illustrative cases: confident synthetic, confident real, and any errors
        picks = pd.concat([
            pr[(pr.true_label == 1) & (pr.pred_label == 1)].nlargest(1, "prob_synthetic"),
            pr[(pr.true_label == 0) & (pr.pred_label == 0)].nsmallest(1, "prob_synthetic"),
            pr[pr.true_label != pr.pred_label].head(2),
        ])
        print(f"Explaining {len(picks)} notes for {TAG}...", flush=True)

        sv = explainer(picks["input"].tolist())

        for k, (_, row) in enumerate(picks.iterrows()):
            html = shap.plots.text(sv[k], display=False)
            fn = (f"/projects/b5bg/results/cbert_tokens_{TAG}_{k}"
                  f"_true{row.true_label}_pred{row.pred_label}.html")
            with open(fn, "w") as fh:
                fh.write(html)
            print(f"  saved {fn}  (P_syn={row.prob_synthetic:.3f})", flush=True)

        del model, clf_pipe, explainer
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"!! {TAG} failed: {e}", flush=True)
        continue