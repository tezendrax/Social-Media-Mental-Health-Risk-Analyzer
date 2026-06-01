"""
train_sentiment.py — 5-class sentiment intensity classifier
Classes: Very Negative, Negative, Neutral, Positive, Very Positive

train_all.py — Master runner for all 4 models
"""
import os, sys, random, joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import clean_text

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

SENTIMENT_DATA = {
    "Very Negative": [
        "I absolutely hate everything about my life right now.",
        "This is the worst situation I have ever been in.",
        "I am completely disgusted and devastated by everything.",
        "Everything is ruined and I feel utterly destroyed.",
        "I despise this situation with every part of me.",
        "Nothing could be worse than what I am going through.",
        "I feel total despair and complete hopelessness.",
        "Life is a nightmare and I cannot endure it anymore.",
        "This is catastrophic. Everything has fallen apart badly.",
        "I am utterly broken and shattered by this pain.",
        "I cannot go on like this. It is absolutely unbearable.",
        "This is rock bottom and it feels completely permanent.",
        "I am consumed by darkness and deep overwhelming despair.",
        "Nothing matters and nothing will ever matter again.",
        "I feel completely destroyed by what has happened.",
        "This pain is the worst I have ever experienced in life.",
        "I cannot imagine surviving this level of suffering.",
        "I am in total agony. There is absolutely no relief.",
        "Everything I valued is completely gone and destroyed.",
        "I feel mortified and devastated beyond all repair.",
    ],
    "Negative": [
        "Things are not going well and I feel pretty bad.",
        "I'm having a rough time and struggling to cope.",
        "I feel sad and disappointed about how things went.",
        "This has been a difficult and draining period for me.",
        "I'm unhappy with how things are currently going.",
        "I feel down and unmotivated most of the time lately.",
        "Things are hard and I'm not doing particularly well.",
        "I'm frustrated and discouraged by recent events.",
        "I feel low and things have not been going my way.",
        "I'm dealing with some difficult and painful emotions.",
        "I'm struggling and things feel heavier than usual.",
        "I feel worried and anxious about several things.",
        "I'm not in a great place right now emotionally.",
        "Things have been tough lately and I feel worn down.",
        "I feel upset and troubled by recent developments.",
        "I'm having trouble finding motivation or energy.",
        "I feel unsatisfied and somewhat pessimistic lately.",
        "Things could definitely be better than they are now.",
        "I'm experiencing some sadness and disappointment.",
        "I feel bothered and concerned about my situation.",
    ],
    "Neutral": [
        "Things are okay. Nothing particularly exciting or bad.",
        "I'm doing fine. Just going about my normal routine.",
        "Life is as expected. Nothing major to report.",
        "I feel neither happy nor sad. Just existing normally.",
        "Things are stable and uneventful. That is fine.",
        "I am okay. No complaints but nothing to celebrate.",
        "Today was ordinary in every sense of the word.",
        "I feel calm and balanced. No strong feelings today.",
        "Everything is proceeding as normal without issues.",
        "I had a regular unremarkable day with nothing notable.",
        "I'm in a neutral headspace. Just going with the flow.",
        "Things are average. Not good, not bad, just average.",
        "I feel indifferent to what is happening around me.",
        "My emotional state is flat and steady today.",
        "No major ups or downs today. Just steady as usual.",
        "I feel fine. Everything is manageable and stable.",
        "Today was nothing special but nothing bad either.",
        "I'm feeling okay. Moderate and stable emotionally.",
        "Things are just the way they are. I am at peace.",
        "I feel balanced and grounded with no strong feelings.",
    ],
    "Positive": [
        "Things are going pretty well and I feel good today.",
        "I'm doing well and feeling optimistic about things.",
        "I feel content and satisfied with how things are.",
        "Today was a good day. I feel positive and hopeful.",
        "I'm in a good mood and things are looking up nicely.",
        "I feel happy and encouraged by recent developments.",
        "Things are improving and I feel genuinely good.",
        "I feel motivated and energized about my situation.",
        "I'm enjoying life and feeling grateful for it.",
        "Today brought some good moments and good feelings.",
        "I feel pleased with how things have been going.",
        "Things are good and I'm feeling optimistic overall.",
        "I feel upbeat and confident about the way things are.",
        "I'm in a great mood and enjoying the positive energy.",
        "Things are coming together nicely. I feel good.",
        "I feel cheerful and satisfied with my current situation.",
        "Today was genuinely enjoyable and I feel happy.",
        "I'm grateful for the positive things happening now.",
        "I feel enthusiastic and hopeful about what is coming.",
        "Things are looking bright and I feel really great.",
    ],
    "Very Positive": [
        "I am absolutely thrilled and ecstatic about everything!",
        "This is the best thing that has ever happened to me!",
        "I feel completely euphoric and overjoyed right now!",
        "Everything is absolutely perfect and I am so happy!",
        "I am on top of the world and nothing can stop me!",
        "This is incredible. I feel like I'm floating on air!",
        "I am bursting with joy and gratitude for this moment!",
        "The most amazing thing happened and I am ecstatic!",
        "I feel absolute pure bliss. Life is completely perfect!",
        "I am completely overwhelmed with happiness and joy!",
        "This is a dream come true. I feel truly blessed!",
        "I have never been this happy. Absolutely incredible!",
        "Pure joy and elation. This is the pinnacle of happiness!",
        "Everything has come together perfectly. I am overjoyed!",
        "I feel so alive, so vibrant, and so incredibly happy!",
        "This is beyond wonderful. I am absolutely thrilled!",
        "I am radiating happiness and positive energy today!",
        "Life is absolutely beautiful and I feel so blessed!",
        "I am exhilarated and grateful beyond all measure!",
        "The most amazing news. I feel completely euphoric!",
    ],
}

PREFIXES = ["", "Honestly: ", "Real talk — ", "Update: "]
SUFFIXES = ["", " That sums it up.", " Needed to express that.", ""]


def build_sentiment_dataset():
    random.seed(42)
    X, y = [], []
    for sentiment, templates in SENTIMENT_DATA.items():
        for t in templates:
            X.append(f"{random.choice(PREFIXES)}{t}{random.choice(SUFFIXES)}".strip())
            y.append(sentiment)
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    return list(X), list(y)


def train_sentiment_model():
    print("=" * 56)
    print("  Sentiment Intensity Classifier — Training")
    print("=" * 56)
    X_raw, y_raw = build_sentiment_dataset()
    print(f"[1/4] Dataset: {len(X_raw)} samples, {len(set(y_raw))} classes")
    X_clean = [clean_text(t) for t in X_raw]
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    print(f"      Classes: {list(le.classes_)}")
    X_tr, X_te, y_tr, y_te = train_test_split(X_clean, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True, min_df=1)),
        ('lr',    LogisticRegression(C=2.0, max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_tr, y_tr)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(pipeline, X_tr, y_tr, cv=cv, scoring='f1_weighted').mean()
    print(f"[3/4] 5-Fold CV F1: {cv_f1:.3f}")
    print(classification_report(y_te, pipeline.predict(X_te), target_names=le.classes_))
    joblib.dump(pipeline, os.path.join(MODELS_DIR, 'sentiment_pipeline.pkl'))
    joblib.dump(le,       os.path.join(MODELS_DIR, 'sentiment_encoder.pkl'))
    print("[4/4] Sentiment model saved.\n")


if __name__ == "__main__":
    train_sentiment_model()
