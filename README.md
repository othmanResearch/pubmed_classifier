# PubMed classifier

Pipeline for PubMed classification used for data curation in AGMP. It collects
biomedical annotations from **PubTator3** (with **BERN2** as a fallback),
preprocesses the annotated abstracts, trains a text classifier, and tracks every
training run with **MLflow**.

## Environment setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Required libraries: python, pandas, numpy, scikit-learn, scipy, metaflow,
mlflow, spacy, requests, pyyaml, tqdm.

## Workflow

Each stage is a standalone Metaflow flow driven by a JSON config in `config/`.
Configs are passed with `--config <name> <path>` before the `run` command.

### 1. Collect annotations (PubTator3 -> BERN2 fallback)

Reads a file of PubMed IDs (one per line) and writes BERN2-shaped JSON chunks.

```powershell
python scripts/collect_annotations.py --config collect config/collect_pubtator.json run
```

PubTator3 is queried first; any PMID it cannot annotate falls back to BERN2.
The normalized output is a drop-in replacement for the existing BERN2 chunks.

### 2. Preprocess NER annotations

Filters by probability, keeps human species, removes overlapping spans, and
inserts inline entity placeholders.

```powershell
python scripts/preprocess_ners.py --config preprocess config/preprocess_postive.json run
python scripts/preprocess_ners.py --config preprocess config/preprocess_negative.json run
```

### 3. Tokenize and clean

```powershell
python scripts/preprocess_annotated_text.py --config preprocess config/preprocess_postive.json run
python scripts/preprocess_annotated_text.py --config preprocess config/preprocess_negative.json run
```

### 4. Remove positive/negative overlap (optional)

```powershell
python scripts/remove_overlapping.py --config overlap config/remove_overlap.json run
```

### 5. Train, validate, evaluate (with MLflow tracking)

Performs a stratified **train / validation / test** split, cross-validates on
the training set, evaluates on the held-out test set, and logs params, metrics
and artifacts to MLflow.

```powershell
python scripts/train_model.py --config training config/train_model.json run
mlflow ui --backend-store-uri ./mlruns      # browse runs at http://127.0.0.1:5000
```

### 6. Predict new PMIDs

```powershell
python scripts/predict_pmid.py --config predict config/predict.json run
```
