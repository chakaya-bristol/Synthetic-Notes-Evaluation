import spacy
import re
import time
import warnings
import os

import pandas as pd

from spacy.symbols import ORTH
from tqdm import tqdm
from pathlib import Path
from spacy.language import Language
from heuristic_tokenize import sent_tokenize_rules

warnings.filterwarnings('ignore')

# Paths
DATA_PATH   = Path("/projects/b5bg/discharge.csv.gz")
OUTPUT_DIR  = Path("/projects/b5bg/processed/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load Data
data = pd.read_csv(DATA_PATH, usecols=["subject_id", "text"])

# spaCy setup
@Language.component("sbd_component")
def sbd_component(doc):
    for i, token in enumerate(doc[:-2]):
        if token.text == '.' and doc[i+1].is_title:
            doc[i+1].sent_start = True
        if token.text == '-' and doc[i+1].text != '-':
            doc[i+1].sent_start = True
    return doc

nlp = spacy.load('en_core_sci_md', disable=['tagger', 'ner'])
nlp.tokenizer.add_special_case("<PHI>", [{ORTH: "<PHI>"}])
nlp.add_pipe("sbd_component", before='parser')

# Helper functions
def fix_deid_tokens(text, processed_text):
    deid_regex = r"\[\*\*.{0,15}.*?\*\*\]"
    if text:
        indexes = [m.span() for m in re.finditer(deid_regex, text, flags=re.IGNORECASE)]
    else:
        indexes = []
    for start, end in indexes:
        processed_text.merge(start_idx=start, end_idx=end)
    return processed_text

def is_structured_block(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    score = 0
    for line in lines:
        clean_line = re.sub(r"\s+", " ", line).strip()
        if clean_line.isupper() and len(clean_line.split()) <= 8:
            score += 1
        if re.search(r"\b(WBC|RBC|Hgb|Hct|MCV|MCH|MCHC|RDW|Plt|Na|K|Cl|HCO3|Creat|Glucose|Calcium|Phos|Mg|PTT|INR)\b", clean_line):
            score += 1
        if re.match(r"^(\d+\.|-|\*)", clean_line):
            score += 1
        if re.match(r"^[A-Za-z /()]+:", clean_line):
            score += 1
    return score >= 2

def process_text(sent, note):
    # sent is now a spaCy Span directly (no DataFrame wrapping)
    sent_text = sent.text
    if not isinstance(sent_text, str) or len(sent_text.strip()) == 0:
        return
    if "\n" in sent_text and is_structured_block(sent_text):
        lines = []
        for line in sent_text.splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            if len(line) > 0:
                lines.append(line)
        note["text"] += "\n".join(lines) + "\n"
    else:
        sent_text = sent_text.replace("\n", " ")
        sent_text = re.sub(r"\s+", " ", sent_text).strip()
        note["text"] += sent_text + "\n"

def process_note_helper(note):
    note = re.sub(r'_{3,}', ' <PHI> ', note)
    note = re.sub(r'(\d+)-\s*\n\s*(\d+)', r'\1-\2', note)
    note_sections = sent_tokenize_rules(note)

    processed_sections = []
    # Use nlp.pipe() for batched spaCy processing (faster than calling nlp() one at a time)
    for section, processed_section in zip(note_sections, nlp.pipe(note_sections)):
        processed_section = fix_deid_tokens(section, processed_section)
        processed_sections.append(processed_section)

    return processed_sections

def process_note(note):
    try:
        note_text = note["text"]
        if pd.isna(note_text):
            note["text"] = ""
            return note

        note_text = str(note_text)
        note["text"] = ""

        processed_sections = process_note_helper(note_text)

        # Plain loops instead of nested DataFrame.apply(axis=1)
        for processed_section in processed_sections:
            for sent in processed_section.sents:
                process_text(sent, note)

        # Post-processing cleanup
        note["text"] = re.sub(r'(\d+)-\s*\n\s*(\d+)', r'\1-\2', note["text"])
        note["text"] = re.sub(r'\n\s*\.\s*\n', '.\n', note["text"])
        note["text"] = re.sub(r'\s+([.,;:?!])', r'\1', note["text"])
        note["text"] = re.sub(r'\n-\n', '\n- ', note["text"])
        note["text"] = re.sub(r'\s+-\s*\n', '\n- ', note["text"])
        note["text"] = re.sub(r'[ \t]+', ' ', note["text"])
        note["text"] = note["text"].strip()

        return note

    except Exception as e:
        print("Error processing note:", e)
        note["text"] = ""
        return note

# Main
if __name__ == "__main__":
    start = time.time()
    tqdm.pandas()

    print('Begin reading notes')
    print('Number of notes: %d' % len(data.index))
    data['ind'] = list(range(len(data.index)))

    formatted_notes = data.progress_apply(process_note, axis=1)

    cleaned_df = formatted_notes.rename(columns={"text": "cleaned_text"})
    cleaned_df = cleaned_df[["subject_id", "cleaned_text"]].copy()
    cleaned_df = cleaned_df.dropna(subset=["cleaned_text"])
    cleaned_df = cleaned_df[cleaned_df["cleaned_text"].str.strip() != ""]

    output_file = OUTPUT_DIR / "discharge_cleaned.csv"
    cleaned_df.to_csv(output_file, index=False)

    end = time.time()
    print(end - start)
    print("Done formatting notes")
    print("Saved to:", output_file)