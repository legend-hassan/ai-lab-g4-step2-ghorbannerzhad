"""Train/evaluate the baseline. Use --data path/to/class-folders."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)
from .features import load_folder_dataset
from .model import build_classifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw", help="root/genre/audio files")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    X, y, _ = load_folder_dataset(args.data)
    counts = {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    if min(counts.values()) < 2:
        raise ValueError("Each genre needs at least two audio files for a stratified split")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y)
    model = build_classifier(args.seed).fit(Xtr, ytr)
    pred = model.predict(Xte)
    labels = list(model.classes_)
    report = classification_report(yte, pred, labels=labels, output_dict=True,
                                   zero_division=0)
    result = {"accuracy": float(accuracy_score(yte, pred)),
              "macro_f1": float(f1_score(yte, pred, average="macro")),
              "class_counts": counts,
              "labels": labels,
              "confusion_matrix": confusion_matrix(yte, pred, labels=labels).tolist(),
              "classification_report": report,
              "split_note": "Stratified random split; replace with artist/album-group split when metadata is available."}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    import joblib
    joblib.dump(model, out / "model.joblib")
    print(json.dumps({"accuracy": result["accuracy"], "macro_f1": result["macro_f1"],
                      "labels": labels}, indent=2))
    print(f"Saved metrics and model in {out.resolve()}")

if __name__ == "__main__":
    main()
