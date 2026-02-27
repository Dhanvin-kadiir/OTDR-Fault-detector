"""Train a Random Forest classifier on engineered OTDR features."""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Ensure the src/ directory is importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import make_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Train fault-detection model")
    ap.add_argument("--data", required=True, help="Path to the multi-trace CSV")
    ap.add_argument("--model_path", default="data/rf_model.pkl", help="Output model pickle")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X, y = make_dataset(df)

    if X.empty:
        print("No features extracted – check your input data.")
        return

    # Check class distribution before training
    class_counts = y.value_counts().sort_index()
    print("Class distribution in training data:")
    print(class_counts.to_string())
    print()

    unique_classes = y.unique()
    if len(unique_classes) < 2:
        print("WARNING: Only one class found! Training data is not balanced – "
              "check synthetic_otdr_generator.py --break_prob setting.")

    # Stratified split; fall back to non-stratified if any class has < 2 samples
    try:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        print("Warning: Stratified split failed (rare class). Using random split.")
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(
        n_estimators=450, random_state=42, class_weight="balanced"
    )
    clf.fit(Xtr, ytr)

    yp = clf.predict(Xte)
    print("Confusion matrix:")
    print(confusion_matrix(yte, yp, labels=sorted(unique_classes)))
    print()
    print(classification_report(yte, yp, zero_division=0))

    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.model_path)
    print(f"[OK] Saved model -> {args.model_path}")
    print(f"     Classes learned: {clf.classes_.tolist()}")


if __name__ == "__main__":
    main()
