"""
Plain-Python demo predictor (runs on Windows; no Metaflow required).

Loads a trained pipeline and classifies either ad-hoc text or a pickled
[texts, pmids] dataset, printing the positive-class probability for each.

Usage:
    # classify a few rows from an already-processed dataset
    python scripts/demo_predict.py --data C:/Users/ghof/AGMP/AGMP/data/pipline_agmp_data/positive.pkl --limit 5

    # classify ad-hoc text (note: not NER-preprocessed, sanity check only)
    python scripts/demo_predict.py --text "GENE__ polymorphism associated with DISEASE risk"
"""
import argparse
import pickle

import joblib

DEFAULT_MODEL = "C:/Users/ghof/pubmed_classifier/output/text_class_predictor.pkl"


def parse_args():
    p = argparse.ArgumentParser(description="Demo prediction with the PubMed classifier")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Path to the trained .pkl pipeline")
    p.add_argument("--data", help="Pickled [texts, pmids] dataset to classify")
    p.add_argument("--text", action="append", help="Ad-hoc text to classify (repeatable)")
    p.add_argument("--limit", type=int, default=10, help="Max rows to classify from --data")
    return p.parse_args()


def main():
    args = parse_args()
    pipeline = joblib.load(args.model)

    if args.data:
        with open(args.data, "rb") as f:
            texts, pmids = pickle.load(f)
        texts, pmids = list(texts)[: args.limit], list(pmids)[: args.limit]
    elif args.text:
        texts, pmids = args.text, [None] * len(args.text)
    else:
        raise SystemExit("provide either --data <pkl> or --text <string>")

    preds = pipeline.predict(texts)
    probs = pipeline.predict_proba(texts)[:, 1]

    print(f"{'pmid':<12} {'pred':<5} {'prob_class1':<11} text")
    for pmid, pred, prob, text in zip(pmids, preds, probs, texts):
        snippet = text[:60].replace("\n", " ")
        print(f"{str(pmid):<12} {int(pred):<5} {prob:<11.3f} {snippet}")


if __name__ == "__main__":
    main()
