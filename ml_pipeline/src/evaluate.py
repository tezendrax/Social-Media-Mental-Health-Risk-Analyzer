"""
evaluate.py
-----------
Load the saved model and run a full evaluation report on a held-out test split.

Usage:
    cd ml_pipeline/src
    python evaluate.py
"""

import os
import sys
import json
import joblib
import numpy as np

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import clean_text
from train import build_dataset

MODELS_DIR         = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH         = os.path.join(MODELS_DIR, 'model_pipeline.pkl')
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')
FEATURE_INFO_PATH  = os.path.join(MODELS_DIR, 'feature_info.json')


def evaluate():
    print("=" * 60)
    print("  Mental Health Risk Classifier — Evaluation Report")
    print("=" * 60)

    # ── Load model artifacts ──────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] Model not found at {MODEL_PATH}")
        print("        Run train.py first.")
        sys.exit(1)

    print("\n[1/3] Loading model artifacts...")
    pipeline = joblib.load(MODEL_PATH)
    le       = joblib.load(LABEL_ENCODER_PATH)
    print(f"      Model loaded from  : {MODEL_PATH}")
    print(f"      Classes            : {list(le.classes_)}")

    # ── Rebuild the same dataset split (same seed) ────────────────────────────
    print("\n[2/3] Rebuilding evaluation dataset...")
    X_raw, y_raw = build_dataset(n_per_class=170, seed=42)
    X_clean = [clean_text(t) for t in X_raw]
    y_enc   = le.transform(y_raw)

    _, X_test, _, y_test = train_test_split(
        X_clean, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"      Test samples       : {len(X_test)}")

    # ── Predict ───────────────────────────────────────────────────────────────
    print("\n[3/3] Running predictions...")
    y_pred      = pipeline.predict(X_test)
    y_prob      = pipeline.predict_proba(X_test)
    accuracy    = accuracy_score(y_test, y_pred)

    print(f"\n  Accuracy : {accuracy:.4f} ({accuracy*100:.1f}%)")
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print("\n".join(f"  {line}" for line in report.split("\n")))

    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    class_names = le.classes_
    header = "       " + "  ".join(f"{c[:8]:>8}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {class_names[i][:8]:>8}  " + "  ".join(f"{v:>8}" for v in row))

    # ROC-AUC (one-vs-rest)
    try:
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
        print(f"\n  ROC-AUC (weighted OVR) : {auc:.4f}")
    except Exception:
        pass

    # ── Top features ──────────────────────────────────────────────────────────
    if os.path.exists(FEATURE_INFO_PATH):
        print("\n  Top Features per Class:")
        print("  " + "-" * 56)
        with open(FEATURE_INFO_PATH) as f:
            feature_info = json.load(f)
        for cls, features in feature_info.items():
            print(f"\n  [{cls}]")
            for item in features[:10]:
                bar = "█" * int(abs(item['weight']) * 15)
                print(f"    {item['feature']:25s}  {item['weight']:+.4f}  {bar}")

    # ── Manual prediction demo ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Sample Predictions Demo")
    print("=" * 60)
    demos = [
        "I can't go on anymore. Everything feels hopeless.",
        "Been a bit stressed and anxious about the upcoming exam.",
        "Had the most amazing weekend with my friends!",
        "I don't know how to cope. I'm so exhausted and alone.",
    ]
    for text in demos:
        cleaned = clean_text(text)
        probs   = pipeline.predict_proba([cleaned])[0]
        pred_i  = np.argmax(probs)
        pred_label = le.inverse_transform([pred_i])[0]
        confidence = probs[pred_i]
        print(f"\n  Input      : {text[:65]}")
        print(f"  Prediction : {pred_label} ({confidence*100:.1f}% confidence)")
        for i, cls in enumerate(le.classes_):
            bar = "█" * int(probs[i] * 20)
            print(f"    {cls:15s}: {probs[i]:.3f}  {bar}")

    print("\n" + "=" * 60)
    print("  Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
