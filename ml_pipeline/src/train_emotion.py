"""
train_emotion.py — 8-class emotion classifier
Emotions: Joy, Sadness, Anxiety, Anger, Fear, Loneliness, Hopelessness, Neutral
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

EMOTION_DATA = {
    "Joy": [
        "I'm so happy today, everything is going great!",
        "Had an amazing time with family, feeling blessed.",
        "Just got great news, I'm over the moon!",
        "Life is beautiful. I feel so grateful and alive.",
        "Best day ever. So much love and laughter.",
        "I feel genuinely happy and content with my life.",
        "My heart is so full right now. Incredible day.",
        "Great news arrived today and I cannot stop smiling.",
        "Pure bliss. This is exactly what happiness feels like.",
        "I woke up feeling so positive and energized today.",
        "Got promoted! I feel amazing and very proud of myself.",
        "Sunny day, good people, great food. Life is wonderful.",
        "I feel alive, vibrant, and full of deep excitement.",
        "Happiness is contagious and today I'm spreading it.",
        "I feel so at peace and happy deep in my soul.",
        "My dreams are coming true and I am overjoyed.",
        "Celebrated a big win with my team. So joyful!",
        "Everything feels perfect right now. Pure joy inside.",
        "I laughed until I cried today. Such a happy feeling.",
        "Grateful for all the good things happening in my life.",
        "I feel loved and supported. So much gratitude today.",
        "Today was wonderful. I feel light and free.",
        "We had a perfect evening together. I feel so happy.",
        "Just had the most joyful experience with old friends.",
        "Everything is working out beautifully. I feel great.",
    ],
    "Sadness": [
        "I feel so sad today. Everything feels grey and empty.",
        "I've been crying all day and I don't know why.",
        "Missing someone so much it physically hurts inside.",
        "I feel a deep sadness that just won't go away.",
        "Nothing feels good anymore. I'm just deeply sad.",
        "Tears keep coming without warning. I feel broken.",
        "I lost something important today. I feel devastated.",
        "Grief is overwhelming me. I can barely breathe right now.",
        "The sadness sits heavy on my chest like a stone.",
        "I feel melancholy and I can't shake this low feeling.",
        "I cry myself to sleep most nights lately.",
        "There is a hollow sadness inside me I cannot explain.",
        "Today I realized I've lost people who mattered most.",
        "Everything reminds me of better times. I'm so sad.",
        "The world feels dim and colorless right now.",
        "I feel grief even when nothing specific happened.",
        "My heart aches all the time and I do not know why.",
        "I feel a profound sadness about the passage of time.",
        "I cannot stop thinking about everything I have lost.",
        "The sadness comes in waves. Today was a very hard wave.",
        "I feel weepy and tender. Everything makes me so sad.",
        "My eyes well up so easily lately. Deep sadness inside.",
        "Something beautiful ended today. I feel very sad.",
        "I feel like crying in public sometimes. Just very sad.",
        "This quiet sadness is my most constant companion.",
    ],
    "Anxiety": [
        "I cannot stop worrying about everything. My mind races.",
        "Panic attack hit me out of nowhere. I am shaking badly.",
        "I'm so anxious about tomorrow that I cannot relax at all.",
        "My heart is pounding and I do not know why.",
        "Overthinking every little thing is exhausting me today.",
        "I feel on edge constantly, like something bad will happen.",
        "Social situations make me want to run and hide.",
        "The anxiety is through the roof. I cannot breathe properly.",
        "My palms are sweaty and my thoughts are racing nonstop.",
        "I dread waking up because the worry starts immediately.",
        "I cannot focus because my anxiety is so loud right now.",
        "Everything feels like a threat. High alert all the time.",
        "I worry about worrying. It is an exhausting cycle.",
        "My body is tense from constant nervous anticipation.",
        "I freeze up whenever I have to do something uncertain.",
        "Every decision feels terrifying because of what might go wrong.",
        "The what-ifs keep spinning in my head without stopping.",
        "I cannot sleep because my mind will not stop planning for disaster.",
        "I am scared of something vague and undefined. Anxious all day.",
        "My stomach is always in knots from constant worry.",
        "I feel hypervigilant and scanning for danger that is not there.",
        "Social anxiety made me cancel my plans again today.",
        "The uncertainty of everything is making me spiral badly.",
        "I keep checking and rechecking things out of anxiety.",
        "Even good things cause me anxiety because of what could go wrong.",
    ],
    "Anger": [
        "I'm furious. I cannot believe what just happened today.",
        "So angry right now. I need to calm down but I cannot.",
        "This is infuriating. I feel like I am about to explode.",
        "I'm outraged by how unfair everything is right now.",
        "I'm seething with anger and trying not to lash out.",
        "Everything is making me irritable and short-tempered today.",
        "I've had enough. This situation makes me so very angry.",
        "I'm enraged and I do not know how to release this feeling.",
        "My blood is boiling. Completely and utterly fed up.",
        "Resentment is building up inside me like pressure.",
        "I cannot control the anger I feel about this situation.",
        "I feel a burning rage that I'm trying hard to keep contained.",
        "This is deeply unfair and I am furious about it.",
        "I feel hostile and aggressive. Not myself right now at all.",
        "Someone crossed a line and now I am absolutely livid.",
        "I'm grinding my teeth because I am so intensely frustrated.",
        "I snapped at someone innocent because I was so angry inside.",
        "The frustration has curdled into genuine burning rage.",
        "I feel betrayed and now I am just pure, hot angry.",
        "I'm mad at the world and I cannot seem to let it go.",
        "Irritability and anger are my constant companions lately.",
        "I feel explosive today. Very close to the edge.",
        "I'm furious at the injustice I witnessed. I want to fight.",
        "Rage is the only emotion I can access right now today.",
        "I feel bitter and angry about how everything turned out.",
    ],
    "Fear": [
        "I'm terrified and I do not know what to do right now.",
        "Something is deeply wrong and I am very scared of it.",
        "I feel a creeping dread about the future closing in.",
        "I'm afraid of what comes next. The uncertainty is terrifying.",
        "Fear has paralyzed me. I cannot make any decisions at all.",
        "I feel scared and vulnerable and very alone in this fear.",
        "I'm frightened by my own thoughts sometimes at night.",
        "A deep sense of dread follows me absolutely everywhere.",
        "I'm scared to fail. The fear is completely crippling me.",
        "The fear of rejection keeps me from trying anything new.",
        "I'm petrified of losing the people I love the most.",
        "Fear of the unknown is keeping me up all night.",
        "I feel unsafe even though I cannot quite explain why.",
        "I feel fearful when I am alone in the dark.",
        "The fear of losing control scares me very deeply.",
        "I'm terrified of the darkness that exists in my own mind.",
        "Fearing the worst has become my absolute default mode.",
        "I'm scared but I do not know of what exactly.",
        "The fear wraps around me like a very cold fog.",
        "I'm afraid of what people really think about me.",
        "The dread of bad news keeps me from checking my phone.",
        "Fear of commitment is slowly ruining my relationships.",
        "I feel afraid and small in a very big threatening world.",
        "I cannot move forward because the fear is simply too strong.",
        "I'm frightened by how quickly things can fall apart.",
    ],
    "Loneliness": [
        "I feel completely alone even in a room full of people.",
        "Nobody really knows me. The loneliness is crushing me.",
        "I have no one to talk to. I feel so deeply isolated.",
        "The silence in my apartment is absolutely suffocating me.",
        "I'm surrounded by people but I feel utterly alone.",
        "I wish someone would just check on me today.",
        "I feel disconnected from everyone and everything around me.",
        "I have no real friends. The isolation is very real.",
        "I could disappear and nobody would notice for days.",
        "Loneliness has become my most constant companion.",
        "I yearn for connection but feel too broken to seek it.",
        "Nobody calls. Nobody texts. There is complete silence.",
        "I feel invisible to everyone I truly care about.",
        "The loneliness hits hardest late at night when I am alone.",
        "I'm an outsider looking in at everyone else's connection.",
        "I've forgotten what it feels like to be truly understood.",
        "Eating alone, sleeping alone, existing alone every single day.",
        "I miss having someone who genuinely cares about me.",
        "My loneliness is a hollow ache that never quite goes away.",
        "I'm alone in a city of millions. Most alone I've ever felt.",
        "I reach out sometimes but people never truly reach back.",
        "No one truly knows what is actually going on inside me.",
        "I long for real connection but feel incapable of it.",
        "I feel like a ghost that absolutely no one can see.",
        "Being alone has started to feel permanent and endless.",
    ],
    "Hopelessness": [
        "I don't see any future for myself. Absolutely none at all.",
        "What is the point of trying? Nothing will ever change.",
        "I've given up hope that things will get any better.",
        "The future looks completely dark and empty to me now.",
        "I cannot imagine ever feeling better than I do now.",
        "Hope is a luxury I can simply no longer afford.",
        "I feel defeated by life. There is no real way forward.",
        "Everything I try fails. I have completely stopped trying.",
        "There is absolutely no light at the end of this tunnel.",
        "I feel like I'm sinking and no one can pull me out.",
        "Nothing will ever change. This is my life forever now.",
        "I've lost all faith in myself, in others, in everything.",
        "The world has nothing left to offer me anymore.",
        "I cannot see a single reason to keep going forward.",
        "I feel trapped with no exit and no hope of finding one.",
        "Hope died in me a very long time ago already.",
        "Every door is closed. Every path leads absolutely nowhere.",
        "I've accepted that things will never ever improve for me.",
        "I've tried everything. Nothing works. I give up completely.",
        "There is simply no hope left inside me at all.",
        "I used to believe things would get better. I don't anymore.",
        "Despair is the only thing that feels honest to me now.",
        "The hopelessness is complete and entirely all-encompassing.",
        "I no longer make plans because I do not believe in tomorrow.",
        "Why even bother? The outcome is always exactly the same.",
    ],
    "Neutral": [
        "Had a regular day. Nothing particularly special happened today.",
        "Went to work, came home. Pretty standard Tuesday honestly.",
        "I did my errands today. Feeling okay, nothing more or less.",
        "Things are fine. No complaints, no real excitement today.",
        "Just another ordinary day. Not bad, not great at all.",
        "Not much to report. Today was completely uneventful.",
        "Feeling neutral. Just existing and going through the motions.",
        "I'm okay. Not amazing, not terrible. Just okay today.",
        "Did the grocery shopping. Made some plans for next week.",
        "Nothing remarkable happened today. A very routine day.",
        "I feel indifferent. Not happy, not sad. Just here existing.",
        "Got through the day. That is really about all I can say.",
        "Normal life, normal feelings. Nothing to write home about.",
        "I feel calm and stable. No strong feelings either way.",
        "Today was boring in a perfectly acceptable sort of way.",
        "Everything is okay. Status quo is being maintained today.",
        "I'm in a steady, unremarkable emotional state right now.",
        "Standard day. Ate, worked, slept. The usual routine.",
        "No particular feelings today. Just going through life.",
        "Things are as they are. I feel fine about it all.",
        "I feel settled and neutral. Nothing too dramatic happening.",
        "Today happened. Tomorrow will too. That is about it.",
        "I completed my to-do list. Feeling adequately fine.",
        "Life is carrying on as normal. No real complaints today.",
        "Not much emotion today, just function and daily routine.",
    ],
}

PREFIXES = ["", "Honestly, ", "Real talk: ", "Not gonna lie, ", "Today: "]
SUFFIXES = ["", " Just venting.", " Needed to say it.", " Anyone relate?"]


def build_emotion_dataset():
    random.seed(42)
    X, y = [], []
    for emotion, templates in EMOTION_DATA.items():
        for t in templates:
            X.append(f"{random.choice(PREFIXES)}{t}{random.choice(SUFFIXES)}".strip())
            y.append(emotion)
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    return list(X), list(y)


def train_emotion_model():
    print("=" * 56)
    print("  Emotion Classifier — Training")
    print("=" * 56)
    X_raw, y_raw = build_emotion_dataset()
    print(f"[1/4] Dataset: {len(X_raw)} samples, {len(set(y_raw))} emotions")
    X_clean = [clean_text(t) for t in X_raw]
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    print(f"      Emotions: {list(le.classes_)}")
    X_tr, X_te, y_tr, y_te = train_test_split(X_clean, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=4000, sublinear_tf=True, min_df=1)),
        ('lr',    LogisticRegression(C=2.0, max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_tr, y_tr)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(pipeline, X_tr, y_tr, cv=cv, scoring='f1_weighted').mean()
    print(f"[3/4] 5-Fold CV F1: {cv_f1:.3f}")
    print(classification_report(y_te, pipeline.predict(X_te), target_names=le.classes_))
    joblib.dump(pipeline, os.path.join(MODELS_DIR, 'emotion_pipeline.pkl'))
    joblib.dump(le,       os.path.join(MODELS_DIR, 'emotion_encoder.pkl'))
    print("[4/4] Emotion model saved.\n")


if __name__ == "__main__":
    train_emotion_model()
