"""
ml_service.py — Full Mental Health Profiling Service
Loads 4 models: Risk, Emotion, Condition, Sentiment
Returns a comprehensive mental health profile from a single text input.
"""
import os, re, sys, json, joblib
import numpy as np
from typing import Optional

ML_SRC = os.path.join(os.path.dirname(__file__), '..', 'ml_pipeline', 'src')
if os.path.isdir(ML_SRC):
    sys.path.insert(0, ML_SRC)

try:
    from preprocessing import clean_text, extract_signals
    _PREP = True
except ImportError:
    _PREP = False

_MDIR = os.path.join(os.path.dirname(__file__), '..', 'ml_pipeline', 'models')

# Sentiment valence scores (for wellbeing computation)
SENTIMENT_VALENCE = {
    "Very Negative": -1.0,
    "Negative":      -0.5,
    "Neutral":        0.0,
    "Positive":       0.5,
    "Very Positive":  1.0,
}

# Risk score map
RISK_SCORE = {"Low Risk": 0.1, "Moderate Risk": 0.5, "High Risk": 0.9}

# Condition colour map (for frontend)
CONDITION_COLORS = {
    "Depression":          "#818cf8",
    "Anxiety Disorder":    "#f59e0b",
    "Burnout":             "#fb923c",
    "Loneliness/Isolation":"#06b6d4",
    "Suicidal Ideation":   "#ef4444",
    "PTSD-like":           "#a78bfa",
    "Bipolar-like":        "#f472b6",
    "Healthy":             "#10b981",
}

EMOTION_ICONS = {
    "Joy":         "😊", "Sadness":     "😢", "Anxiety":      "😰",
    "Anger":       "😠", "Fear":        "😨", "Loneliness":   "🫂",
    "Hopelessness":"😞", "Neutral":     "😐",
}

# Heuristic fallback keywords
_HIGH_KW = ['suicidal','hopeless','worthless','depressed','give up','cant go on','die','dead inside']
_MOD_KW  = ['anxious','stressed','overwhelmed','sad','lonely','exhausted','tired']


class MLService:
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.feature_info = None
        self._load_all()

    def _load(self, key, pipeline_file, encoder_file):
        pp = os.path.join(_MDIR, pipeline_file)
        ep = os.path.join(_MDIR, encoder_file)
        if os.path.exists(pp) and os.path.exists(ep):
            self.models[key]   = joblib.load(pp)
            self.encoders[key] = joblib.load(ep)
            print(f"[MLService] [OK] {key} model loaded")
            return True
        print(f"[MLService] [WARN] {key} model not found — will use fallback")
        return False

    def _load_all(self):
        self._load('risk',      'model_pipeline.pkl',     'label_encoder.pkl')
        self._load('emotion',   'emotion_pipeline.pkl',   'emotion_encoder.pkl')
        self._load('condition', 'condition_pipeline.pkl', 'condition_encoder.pkl')
        self._load('sentiment', 'sentiment_pipeline.pkl', 'sentiment_encoder.pkl')
        fi = os.path.join(_MDIR, 'feature_info.json')
        if os.path.exists(fi):
            with open(fi) as f:
                self.feature_info = json.load(f)

    @property
    def model_loaded(self):
        return 'risk' in self.models

    def _clean(self, text):
        if _PREP:
            return clean_text(text)
        t = text.lower()
        t = re.sub(r'https?://\S+', '', t)
        t = re.sub(r'[^a-z\s]', '', t)
        return re.sub(r'\s+', ' ', t).strip()

    def _predict_model(self, key, text):
        """Run a model and return (label, confidence, full_proba_dict)."""
        cleaned = self._clean(text)
        mdl = self.models[key]
        enc = self.encoders[key]
        probs = mdl.predict_proba([cleaned])[0]
        idx   = int(np.argmax(probs))
        label = enc.classes_[idx]
        conf  = float(probs[idx])
        proba = {cls: round(float(p), 4) for cls, p in zip(enc.classes_, probs)}
        return label, conf, proba

    # ── Risk ──────────────────────────────────────────────────────────────────
    def _risk_result(self, text):
        if 'risk' not in self.models:
            return self._heuristic_risk(text)
        cls, conf, proba = self._predict_model('risk', text)
        hr  = proba.get("High Risk", 0)
        mr  = proba.get("Moderate Risk", 0)
        gauge = round(hr + 0.5 * mr, 3)
        signals = self._keyword_signals(self._clean(text), cls)
        return {"classification": cls, "probability": gauge,
                "confidence": round(conf, 3), "probabilities": proba,
                "signals": signals, "model_used": "tfidf_logreg"}

    def _heuristic_risk(self, text):
        t = text.lower(); base = 0.08; sigs = []
        for kw in _HIGH_KW:
            if kw in t: base += 0.18; sigs.append({"keyword":kw,"contribution":0.18,"risk_level":"high"})
        for kw in _MOD_KW:
            if kw in t: base += 0.07; sigs.append({"keyword":kw,"contribution":0.07,"risk_level":"moderate"})
        prob = min(base + np.random.uniform(0,.1), 0.97)
        cls  = "High Risk" if prob > 0.6 else "Moderate Risk" if prob > 0.3 else "Low Risk"
        return {"classification": cls, "probability": round(prob,3),
                "confidence": 0.68, "probabilities": {"High Risk":round(prob,3),"Moderate Risk":0.2,"Low Risk":0.1},
                "signals": sigs[:6], "model_used": "heuristic_fallback"}

    def _keyword_signals(self, cleaned, cls):
        if not self.feature_info: return []
        wmap = {i["feature"]: i["weight"] for i in self.feature_info.get(cls, [])}
        tokens = set(cleaned.split())
        words  = cleaned.split()
        bigrams = {f"{words[i]} {words[i+1]}" for i in range(len(words)-1)}
        matched = [{"keyword":t,"contribution":round(abs(wmap[t]),4),"risk_level":cls.lower().replace(" ","_")}
                   for t in (tokens | bigrams) if t in wmap]
        return sorted(matched, key=lambda x: x["contribution"], reverse=True)[:8]

    # ── Emotion ───────────────────────────────────────────────────────────────
    def _emotion_result(self, text):
        if 'emotion' not in self.models:
            return {"dominant": "Neutral", "confidence": 0.5, "icon": "😐",
                    "distribution": {}, "model_used": "unavailable"}
        cls, conf, proba = self._predict_model('emotion', text)
        # Top 3 emotions
        top3 = sorted(proba.items(), key=lambda x: x[1], reverse=True)[:3]
        return {
            "dominant":     cls,
            "confidence":   round(conf, 3),
            "icon":         EMOTION_ICONS.get(cls, "❓"),
            "distribution": proba,
            "top3":         [{"emotion": e, "probability": p, "icon": EMOTION_ICONS.get(e,"❓")} for e, p in top3],
            "model_used":   "tfidf_logreg",
        }

    # ── Condition ─────────────────────────────────────────────────────────────
    def _condition_result(self, text):
        if 'condition' not in self.models:
            return {"primary": "Unknown", "confidence": 0.5,
                    "profile": [], "model_used": "unavailable"}
        cls, conf, proba = self._predict_model('condition', text)
        top3 = sorted(proba.items(), key=lambda x: x[1], reverse=True)[:3]
        profile = [{"condition": c, "probability": round(p, 3),
                    "color": CONDITION_COLORS.get(c, "#94a3b8")} for c, p in top3]
        return {
            "primary":    cls,
            "confidence": round(conf, 3),
            "color":      CONDITION_COLORS.get(cls, "#94a3b8"),
            "profile":    profile,
            "model_used": "tfidf_logreg",
        }

    # ── Sentiment ─────────────────────────────────────────────────────────────
    def _sentiment_result(self, text):
        if 'sentiment' not in self.models:
            return {"label": "Neutral", "valence": 0.0, "confidence": 0.5,
                    "distribution": {}, "model_used": "unavailable"}
        cls, conf, proba = self._predict_model('sentiment', text)
        return {
            "label":        cls,
            "valence":      SENTIMENT_VALENCE.get(cls, 0.0),
            "confidence":   round(conf, 3),
            "distribution": proba,
            "model_used":   "tfidf_logreg",
        }

    # ── Wellbeing Score (composite 0–100) ─────────────────────────────────────
    @staticmethod
    def _compute_wellbeing(risk_prob, sentiment_valence, condition_primary, emotion_dominant):
        # Base from sentiment valence (-1 to 1) → 0 to 100
        base = (sentiment_valence + 1) / 2 * 100
        # Risk penalty
        risk_pen = risk_prob * 45
        # Condition adjustment
        cond_pen = {"Suicidal Ideation": 35, "Depression": 25, "Bipolar-like": 20,
                    "PTSD-like": 18, "Anxiety Disorder": 15, "Burnout": 12,
                    "Loneliness/Isolation": 10, "Healthy": 0}.get(condition_primary, 10)
        # Emotion adjustment
        emo_adj = {"Joy": +12, "Neutral": 0, "Sadness": -8, "Anxiety": -10,
                   "Anger": -8, "Fear": -10, "Loneliness": -10,
                   "Hopelessness": -18}.get(emotion_dominant, 0)
        score = base - risk_pen - cond_pen + emo_adj
        return max(2, min(98, round(score)))

    # ── Public API ────────────────────────────────────────────────────────────
    def profile_text(self, text: str) -> dict:
        """
        Full mental health profile from a single text input.
        Returns: risk, emotion, condition, sentiment, wellbeing_score, signals.
        """
        risk      = self._risk_result(text)
        emotion   = self._emotion_result(text)
        condition = self._condition_result(text)
        sentiment = self._sentiment_result(text)
        wellbeing = self._compute_wellbeing(
            risk_prob           = risk["probability"],
            sentiment_valence   = sentiment["valence"],
            condition_primary   = condition["primary"],
            emotion_dominant    = emotion["dominant"],
        )
        return {
            "risk":       risk,
            "emotion":    emotion,
            "condition":  condition,
            "sentiment":  sentiment,
            "wellbeing_score": wellbeing,
        }

    # Keep legacy predict_risk for backwards compat
    def predict_risk(self, text: str) -> dict:
        return self._risk_result(text)

    def get_signals_for_text(self, text: str) -> list:
        return self._keyword_signals(self._clean(text),
                                     self._risk_result(text)["classification"])

    def get_status(self) -> dict:
        return {
            "model_loaded": self.model_loaded,
            "models_available": list(self.models.keys()),
            "mode": "full_profile" if len(self.models) == 4 else
                    ("partial" if self.models else "heuristic_fallback"),
        }


ml_service = MLService()
