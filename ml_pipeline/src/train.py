"""
train.py
--------
Trains a TF-IDF + Logistic Regression mental health risk classifier on a
realistic synthetic dataset (500 samples, 3 classes).

Usage:
    cd ml_pipeline/src
    python train.py
"""

import os
import sys
import json
import random
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

# Make sure we can import our preprocessing module regardless of CWD
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import clean_text

# ── Output paths ─────────────────────────────────────────────────────────────
MODELS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

MODEL_PATH          = os.path.join(MODELS_DIR, 'model_pipeline.pkl')
LABEL_ENCODER_PATH  = os.path.join(MODELS_DIR, 'label_encoder.pkl')
FEATURE_INFO_PATH   = os.path.join(MODELS_DIR, 'feature_info.json')

# -- Synthetic dataset templates ----------------------------------------------
HIGH_RISK_TEMPLATES = [
    "I can't go on anymore. Everything feels completely hopeless.",
    "I've been feeling so depressed lately, I don't see the point of anything.",
    "I hate myself so much. I don't deserve to live.",
    "The pain is unbearable. I just want it to stop forever.",
    "I've been thinking about ending it all. I'm done fighting.",
    "Nobody would even notice if I disappeared. I'm worthless.",
    "I feel completely empty inside. Life has no meaning for me anymore.",
    "Every morning I wake up and wish I hadn't. This darkness never ends.",
    "I'm so exhausted from existing. I can't keep pretending I'm okay.",
    "I've been crying for days straight. I just want to disappear.",
    "There's no way out of this pain. I've given up hope completely.",
    "I feel like a burden to everyone around me. Better off gone.",
    "I used to have reasons to live. Now there's just nothing left.",
    "The thought of self harm keeps coming back. I'm scared of myself.",
    "I wrote a goodbye note today. I'm so tired of everything.",
    "Suicidal thoughts won't leave me alone. I'm terrified.",
    "I've stopped eating and sleeping. My body just feels dead.",
    "I don't feel real anymore. Like I'm watching myself from far away.",
    "My depression has swallowed me completely. I see no future.",
    "I told someone I was fine today. I wasn't. I haven't been in months.",
    "Relapsed again. I thought I was getting better but I'm back to nothing.",
    "I've isolated myself from everyone. I'm completely alone in this.",
    "The darkness is overwhelming. I'm not strong enough to fight anymore.",
    "I keep imagining what it would feel like to just not exist.",
    "I've been self harming again. I don't know how to cope otherwise.",
    "Everything I do turns to failure. I'm a complete disappointment.",
    "My mind won't stop torturing me with dark thoughts. Please make it stop.",
    "I've lost the ability to feel joy. Everything is just grey emptiness.",
    "I haven't left my bed in a week. I just can't face the world.",
    "Every day feels like a countdown to something terrible happening.",
    "I feel so trapped. Like there's no escape from this pain.",
    "I don't trust myself right now. The thoughts are getting louder.",
    "I've been hiding how bad it is. Nobody knows how close I am to the edge.",
    "Life feels like a cruel joke I was never meant to survive.",
    "I can't remember the last time I felt anything but despair.",
]

MODERATE_RISK_TEMPLATES = [
    "I've been feeling really anxious and stressed lately, struggling to cope.",
    "Work is overwhelming me. I feel exhausted all the time and can't sleep.",
    "I've been having panic attacks more frequently this past week.",
    "I don't know how to deal with all this pressure. I'm falling apart.",
    "Feeling really down and lonely today. Just needed to say it somewhere.",
    "My anxiety has been through the roof. Simple tasks feel impossible.",
    "I'm so burned out. I've lost motivation for everything I used to enjoy.",
    "I've been crying a lot lately without really knowing why.",
    "Social situations have been really hard. I feel invisible to people.",
    "I'm constantly worrying about everything. My mind never quiets down.",
    "The stress is getting to me. I feel like I'm on the edge of breaking.",
    "I've been isolating more lately. Hard to reach out when feeling this way.",
    "Sleep is a mess. Nightmares and racing thoughts keep waking me up.",
    "I feel disconnected from everyone around me lately.",
    "My mental health isn't great right now. Taking it one day at a time.",
    "Struggling with some difficult emotions this week. Being kind to myself.",
    "I've been feeling really sad and can't pinpoint the reason why.",
    "Anxious about everything lately. Even small decisions feel huge.",
    "I just want to hide from the world for a while. Everything feels heavy.",
    "Having a really rough few weeks. My mood has been really low.",
    "I feel like no one truly understands what I'm going through.",
    "Getting through each day feels harder than the last right now.",
    "I don't feel like myself lately. Hoping it passes soon.",
    "My energy is completely drained. Hard to find the motivation to do anything.",
    "Feeling really overwhelmed by life circumstances right now.",
    "I've been avoiding things that normally make me happy. Not sure why.",
    "Life just feels heavy and grey lately. Struggling to find brightness.",
    "Relationship stress has been really taking a toll on my mental wellbeing.",
    "I feel like I'm barely keeping it together. Trying to stay afloat.",
    "The weight of everything has been crushing me slowly this month.",
    "Hard day. Feeling emotional and a bit lost about where my life is going.",
    "I've been more irritable and emotional than usual. Not feeling great.",
    "Spending too much time in my own head with negative thoughts lately.",
    "I feel drained from pretending to be okay when I'm really not.",
    "Some days getting out of bed is genuinely the hardest thing I do.",
]

LOW_RISK_TEMPLATES = [
    "Had an amazing day hiking with my friends today. Feeling so grateful!",
    "Just finished a great book. Life is feeling pretty good right now.",
    "Excited about my new project at work. Can't wait to see it grow.",
    "Spent the afternoon cooking with family. These moments are everything.",
    "Finally got some rest after a busy week. Feeling recharged and happy.",
    "Had coffee with an old friend today. So good to reconnect.",
    "Started learning guitar and I'm absolutely loving it.",
    "The sunrise this morning was breathtaking. Grateful to be alive.",
    "Just hit a personal record at the gym. Really proud of my progress.",
    "Feeling motivated and energized today. Ready to take on challenges.",
    "A quiet evening with a good movie and some hot tea. Perfect.",
    "My dog is the cutest thing. He makes every day brighter.",
    "Feeling really positive about the future. Excited for what's coming.",
    "Got great news today! Dreams really do come true if you work for them.",
    "Baked bread for the first time. It turned out incredible!",
    "Grateful for a supportive team at work. We had a great meeting today.",
    "Nature walk this morning cleared my head. Ready for a productive day.",
    "Feeling content and at peace. Sometimes ordinary days are the best.",
    "Just booked a vacation. Thrilled to have something fun to look forward to.",
    "Finished a project I've been working on. Feeling accomplished!",
    "Had a really meaningful conversation with a close friend today.",
    "The weather is perfect today. Going to enjoy every minute of it.",
    "Feeling proud of how far I've come this year. Growth is real.",
    "A little bit of self-care goes a long way. Feeling refreshed.",
    "Great workout this morning. Mind and body feeling balanced and strong.",
    "Life is good. Not perfect, but genuinely good and I'm thankful for it.",
    "Laughed until I cried with my family tonight. The best kind of tired.",
    "Started journaling again and it's been really positive for my mindset.",
    "Small wins matter. Celebrated finishing a task I'd been putting off.",
    "Feeling inspired after watching a documentary about resilience.",
    "Made a new friend at a community event. Love when that happens.",
    "Cooking a new recipe tonight. Love trying new things in the kitchen.",
    "The long weekend was exactly what I needed. Feeling refreshed and ready.",
    "Went to a local market today. Simple pleasures are truly the best.",
    "Feeling optimistic about next week. Good things are on the horizon.",
]

AUGMENTATION_PREFIXES = [
    "", "Honestly, ", "I just need to say: ", "Real talk — ",
    "Not gonna lie, ", "Been thinking a lot: ", "Day {n}: ",
    "Update: ", "Feeling like: ", "Can't sleep so ", "3am thoughts: ",
]

AUGMENTATION_SUFFIXES = [
    "", " Just venting.", " Needed to get that out.",
    " Anyone else feel this way?", " Thanks for listening.",
    " Is this normal?", " Not sure what to do.",
    " Sending love to anyone else struggling.",
    " I'll be okay, I think.", " This is hard.",
]


def build_dataset(n_per_class: int = 170, seed: int = 42) -> tuple:
    """
    Build a balanced synthetic dataset by augmenting templates.

    Returns:
        X (list of str), y (list of str label)
    """
    random.seed(seed)
    X, y = [], []

    class_map = {
        "High Risk":     HIGH_RISK_TEMPLATES,
        "Moderate Risk": MODERATE_RISK_TEMPLATES,
        "Low Risk":      LOW_RISK_TEMPLATES,
    }

    for label, templates in class_map.items():
        for i in range(n_per_class):
            base = random.choice(templates)
            prefix = random.choice(AUGMENTATION_PREFIXES).replace("{n}", str(i+1))
            suffix = random.choice(AUGMENTATION_SUFFIXES)
            text = f"{prefix}{base}{suffix}".strip()
            X.append(text)
            y.append(label)

    # Shuffle
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    return list(X), list(y)


def get_top_features(pipeline: Pipeline, label_encoder: LabelEncoder,
                     n_top: int = 15) -> dict:
    """
    Extract top TF-IDF feature weights per class from the trained LogReg model.

    Returns:
        Dict mapping class label → list of (feature, weight) tuples
    """
    vectorizer: TfidfVectorizer = pipeline.named_steps['tfidf']
    classifier: LogisticRegression = pipeline.named_steps['lr']
    feature_names = vectorizer.get_feature_names_out()
    top_features = {}

    for i, cls in enumerate(label_encoder.classes_):
        coefs = classifier.coef_[i]
        top_idx = np.argsort(coefs)[::-1][:n_top]
        top_features[cls] = [
            {"feature": feature_names[j], "weight": float(round(coefs[j], 4))}
            for j in top_idx
        ]
    return top_features


def train():
    print("=" * 60)
    print("  Mental Health Risk Classifier — Training Pipeline")
    print("=" * 60)

    # ── 1. Build dataset ──────────────────────────────────────────────────────
    print("\n[1/5] Building synthetic dataset...")
    X_raw, y_raw = build_dataset(n_per_class=170)
    print(f"      Total samples: {len(X_raw)}")
    for lbl in ["High Risk", "Moderate Risk", "Low Risk"]:
        print(f"      {lbl}: {y_raw.count(lbl)}")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    print("\n[2/5] Preprocessing text...")
    X_clean = [clean_text(t) for t in X_raw]

    # ── 3. Encode labels ──────────────────────────────────────────────────────
    print("\n[3/5] Encoding labels...")
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    print(f"      Classes: {list(le.classes_)}")

    # ── 4. Train / Test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"      Train: {len(X_train)} | Test: {len(X_test)}")

    # ── 5. Build & train pipeline ─────────────────────────────────────────────
    print("\n[4/5] Training TF-IDF + Logistic Regression pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            min_df=2,
        )),
        ('lr', LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            multi_class='multinomial',
            solver='lbfgs',
        ))
    ])

    pipeline.fit(X_train, y_train)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_weighted')
    print(f"      5-Fold CV F1 (weighted): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── 6. Evaluate on held-out test set ──────────────────────────────────────
    print("\n[5/5] Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    print("\n" + classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # ── 7. Save artifacts ─────────────────────────────────────────────────────
    joblib.dump(pipeline,  MODEL_PATH)
    joblib.dump(le,         LABEL_ENCODER_PATH)
    print(f"\n[OK] Model saved   -> {MODEL_PATH}")
    print(f"[OK] Encoder saved -> {LABEL_ENCODER_PATH}")

    # Save feature importance info
    top_features = get_top_features(pipeline, le)
    with open(FEATURE_INFO_PATH, 'w') as f:
        json.dump(top_features, f, indent=2)
    print(f"[OK] Feature info  -> {FEATURE_INFO_PATH}")

    print("\n" + "=" * 60)
    print("  Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    train()
