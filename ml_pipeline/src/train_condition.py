"""
train_condition.py — 8-class mental health condition classifier
Conditions: Depression, Anxiety Disorder, Burnout, Loneliness/Isolation,
            Suicidal Ideation, PTSD-like, Bipolar-like, Healthy
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

CONDITION_DATA = {
    "Depression": [
        "I wake up every day feeling empty and grey inside.",
        "I've lost interest in things I used to love doing.",
        "Getting out of bed is the hardest thing I do each day.",
        "I feel worthless and like a burden to everyone around me.",
        "I've been sleeping too much but never feeling rested.",
        "Food has no taste and nothing brings me pleasure anymore.",
        "I feel persistently sad for weeks with no clear reason.",
        "I've stopped caring about my appearance and hygiene.",
        "My concentration is completely shot. I can't focus.",
        "I feel slowed down, like moving through thick fog.",
        "Everything feels pointless and I have no motivation.",
        "I've withdrawn from all social activities and friends.",
        "I feel guilty about everything, even things not my fault.",
        "My energy is completely depleted every single day.",
        "I've been crying for no reason almost every day.",
        "I feel like a failure no matter what I accomplish.",
        "The depression makes even small tasks feel impossible.",
        "I've lost all hope that things will ever get better.",
        "I feel disconnected from reality and from myself.",
        "I have recurring thoughts about death and dying.",
        "I no longer find joy in activities I used to love.",
        "Social isolation has become my comfortable default.",
        "My mood is consistently low. Nothing lifts it anymore.",
        "I feel empty inside, like something important is missing.",
        "Depression is stealing months of my life at a time.",
    ],
    "Anxiety Disorder": [
        "My anxiety is constant and completely overwhelming.",
        "I have panic attacks regularly and they terrify me.",
        "I avoid situations that trigger my anxiety altogether.",
        "I worry excessively about things I cannot control.",
        "My body is in constant fight-or-flight mode always.",
        "I have intrusive thoughts that I cannot stop or control.",
        "Social anxiety prevents me from living a normal life.",
        "I overthink every conversation and social interaction.",
        "My anxiety is physical: tight chest, racing heart always.",
        "I spiral into worst-case scenarios automatically.",
        "I need everything perfectly controlled or I panic.",
        "Generalized worry consumes most of my waking hours.",
        "I'm terrified of judgment and criticism from others.",
        "I check and recheck things obsessively due to anxiety.",
        "My nervous system is constantly on high alert mode.",
        "Uncertainty sends me into an immediate anxiety spiral.",
        "I experience anticipatory anxiety about everything.",
        "Phobias are limiting my daily functioning severely.",
        "I feel restless and on edge virtually all the time.",
        "My anxiety tells me catastrophe is always imminent.",
        "I avoid crowds, new places, and unpredictable situations.",
        "Health anxiety makes me convinced I am always ill.",
        "I have intrusive worries that are hard to dismiss.",
        "My anxiety hijacks rational thinking constantly.",
        "I've been prescribed medication for my anxiety disorder.",
    ],
    "Burnout": [
        "I'm completely exhausted and drained by my work.",
        "I have no energy left for anything after my job.",
        "I feel cynical and detached about my work lately.",
        "I'm emotionally numb and running on empty fumes.",
        "I used to love my job. Now I dread Monday mornings.",
        "I feel like a machine just going through the motions.",
        "I've lost all passion and sense of purpose in my work.",
        "Work feels meaningless and I feel invisible and depleted.",
        "I'm physically and mentally exhausted all of the time.",
        "I can't set boundaries and now I'm completely burnt out.",
        "I feel like I'm giving everything and getting nothing.",
        "My productivity has collapsed along with my wellbeing.",
        "I feel like a shell of the motivated person I once was.",
        "The work never stops and I'm drowning in it all.",
        "I've stopped caring about outcomes. I'm too depleted.",
        "Burnout has left me physically sick and emotionally dead.",
        "I get home and cannot do anything. Total exhaustion.",
        "I feel trapped in a job that is slowly destroying me.",
        "No amount of rest makes the tiredness go away.",
        "I've lost my professional identity in this exhaustion.",
        "I'm resentful of demands placed on me constantly.",
        "My creativity and enthusiasm have completely evaporated.",
        "I feel like a zombie going through professional motions.",
        "Everything about work fills me with dread and exhaustion.",
        "I'm burned out and I don't know how to recover.",
    ],
    "Loneliness/Isolation": [
        "I have no meaningful connections in my life anymore.",
        "I'm surrounded by people but feel utterly isolated.",
        "I've been completely alone for months now.",
        "No one truly understands what I'm going through.",
        "I feel invisible and unimportant to everyone around me.",
        "Social isolation has become my new permanent normal.",
        "I have no one to call when things go wrong.",
        "I'm chronically lonely despite living in a big city.",
        "My social circle has shrunk to almost nothing.",
        "I feel like an outsider in every social situation.",
        "I've forgotten how to connect with people properly.",
        "The isolation is starting to affect my mental health.",
        "I go days without meaningful human conversation.",
        "I feel disconnected from humanity in a deep way.",
        "I eat alone, sleep alone, and exist alone every day.",
        "Loneliness has become my most familiar companion.",
        "I long for connection but cannot seem to find it.",
        "I feel like no one would notice if I disappeared.",
        "I've withdrawn so far inward that I'm lost in myself.",
        "The silence of my life is deafening and suffocating.",
        "I feel estranged from family, friends, and community.",
        "Chronic loneliness is affecting my physical health too.",
        "I yearn for deep friendship but cannot form one.",
        "I feel fundamentally separate from other human beings.",
        "No one truly sees me or knows the real me.",
    ],
    "Suicidal Ideation": [
        "I've been having thoughts about ending my life.",
        "I don't want to exist anymore. The pain is too much.",
        "I've thought about how I would do it. I'm scared.",
        "I feel like everyone would be better off without me.",
        "I've been thinking about death constantly lately.",
        "I wrote something like a goodbye letter last night.",
        "I can't see any reason to keep living right now.",
        "The thought of not existing anymore is comforting.",
        "I've been researching methods and that scares me.",
        "I told someone I was fine but I'm not. Not at all.",
        "I've started giving away things that matter to me.",
        "I feel a strange calm now that I've made a decision.",
        "I'm in so much pain I just want it to stop forever.",
        "I've been thinking about suicide more than I let on.",
        "I don't see a future for myself. I just want peace.",
        "I feel like a burden and the world is better without me.",
        "I've had passive thoughts of death becoming more active.",
        "Something in me is drawn to the idea of not existing.",
        "I've talked to a crisis line before. I'm struggling again.",
        "I'm hanging on by a very thin thread right now.",
        "I feel trapped and suicide seems like the only exit.",
        "I've been self-harming as a way to cope with the pain.",
        "The darkness is so complete that death feels like light.",
        "I'm not safe right now and I know I need help.",
        "I've made a plan and I'm scared of myself.",
    ],
    "PTSD-like": [
        "I keep having flashbacks to the traumatic event.",
        "Certain sounds and smells trigger intense panic in me.",
        "I relive the trauma as if it's happening right now.",
        "I avoid anything that reminds me of what happened.",
        "I have nightmares about the event almost every night.",
        "I feel emotionally numb most of the time after the trauma.",
        "I startle very easily and feel hypervigilant constantly.",
        "I feel detached from my body and from my own life.",
        "I've lost my sense of safety and trust in the world.",
        "The trauma has fundamentally changed who I am.",
        "I feel shame and guilt about something that happened.",
        "I can't talk about it without completely falling apart.",
        "I've been avoiding people, places, and memories related to it.",
        "I feel perpetually on guard as if danger is always near.",
        "I'm easily angered and irritable since the incident.",
        "Intrusive memories interrupt my daily life constantly.",
        "I dissociate when I encounter triggers related to trauma.",
        "I've been told I have trauma responses to normal things.",
        "The trauma shattered my sense of self and safety.",
        "I feel like I'm stuck in the past and can't move on.",
        "Physical sensations trigger the trauma response in me.",
        "I feel hyperaroused and can't fully relax or sleep.",
        "I've lost trust in people after what was done to me.",
        "The trauma surfaces when I least expect or want it to.",
        "I feel broken by what happened to me.",
    ],
    "Bipolar-like": [
        "My moods swing wildly from extreme highs to deep lows.",
        "I was on top of the world last week. Now I'm in a pit.",
        "I went on a huge spending spree during my high phase.",
        "I barely slept for days and felt invincible and amazing.",
        "The crashes after my highs are absolutely devastating.",
        "I made impulsive decisions during my last elevated mood.",
        "My energy goes from zero to infinite with no middle ground.",
        "I feel grandiose and powerful then crash into depression.",
        "People tell me I'm a completely different person in highs.",
        "Racing thoughts and endless energy then total collapse.",
        "I feel invincible then suddenly worthless. No in-between.",
        "My relationships suffer because of my unpredictable moods.",
        "During my highs I barely need sleep and feel euphoric.",
        "During my lows I can barely function or get out of bed.",
        "The mood swings are getting more extreme and rapid.",
        "I made a huge life decision while manic. I regret it.",
        "My high moods feel amazing but always lead to crashes.",
        "I feel like two completely different people inside.",
        "I was hypersexual and reckless during my last episode.",
        "The depression after a manic phase is absolutely brutal.",
        "I oscillate between feeling godlike and feeling worthless.",
        "I've been hospitalized during an extreme mood episode.",
        "My moods cycle in ways I cannot predict or control.",
        "I have periods of extreme creativity then total shutdown.",
        "I've been told I might have bipolar disorder.",
    ],
    "Healthy": [
        "I feel emotionally stable and generally content with life.",
        "I have a good support system and feel genuinely connected.",
        "I'm managing stress effectively and taking good care.",
        "Life has its challenges but I'm coping well overall.",
        "I feel grounded and at peace with where I am in life.",
        "I have healthy relationships and I feel truly appreciated.",
        "I'm able to regulate my emotions without feeling overwhelmed.",
        "I seek help when I need it and that feels empowering.",
        "I feel purposeful and energized by the life I'm living.",
        "My mental health is something I actively work to maintain.",
        "I have good boundaries and communicate my needs clearly.",
        "I experience normal stress but recover from it quickly.",
        "I feel hopeful about the future and excited for what's next.",
        "I practice self-care regularly and it makes a real difference.",
        "I feel connected to my values and live authentically.",
        "I'm resilient and can bounce back from life's setbacks.",
        "I feel satisfaction in my daily life most of the time.",
        "I have a growth mindset and embrace life's challenges.",
        "I feel comfortable with uncertainty about the future.",
        "I'm able to enjoy the present moment and feel gratitude.",
        "I process difficult emotions and let them go healthily.",
        "I feel confident in my ability to handle challenges.",
        "I have meaning, purpose, and a sense of belonging.",
        "My overall wellbeing is good and I'm proud of that.",
        "I feel whole and integrated as a person in my life.",
    ],
}

PREFIXES = ["", "Honestly, ", "Real talk — ", "Update: ", "Day log: "]
SUFFIXES = ["", " Just sharing.", " Needed to vent.", " Any thoughts?"]


def build_condition_dataset():
    random.seed(42)
    X, y = [], []
    for cond, templates in CONDITION_DATA.items():
        for t in templates:
            X.append(f"{random.choice(PREFIXES)}{t}{random.choice(SUFFIXES)}".strip())
            y.append(cond)
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    return list(X), list(y)


def train_condition_model():
    print("=" * 56)
    print("  Condition Classifier — Training")
    print("=" * 56)
    X_raw, y_raw = build_condition_dataset()
    print(f"[1/4] Dataset: {len(X_raw)} samples, {len(set(y_raw))} conditions")
    X_clean = [clean_text(t) for t in X_raw]
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    print(f"      Conditions: {list(le.classes_)}")
    X_tr, X_te, y_tr, y_te = train_test_split(X_clean, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True, min_df=1)),
        ('lr',    LogisticRegression(C=1.5, max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_tr, y_tr)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(pipeline, X_tr, y_tr, cv=cv, scoring='f1_weighted').mean()
    print(f"[3/4] 5-Fold CV F1: {cv_f1:.3f}")
    print(classification_report(y_te, pipeline.predict(X_te), target_names=le.classes_))
    joblib.dump(pipeline, os.path.join(MODELS_DIR, 'condition_pipeline.pkl'))
    joblib.dump(le,       os.path.join(MODELS_DIR, 'condition_encoder.pkl'))
    print("[4/4] Condition model saved.\n")


if __name__ == "__main__":
    train_condition_model()
