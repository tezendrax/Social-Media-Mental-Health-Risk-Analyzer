"""
train_all.py — Master training runner
Trains all 4 models in sequence: Risk, Emotion, Condition, Sentiment
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

print("\n" + "=" * 60)
print("  SUJHAAV INTELLIGENCE — Full Model Training Suite")
print("=" * 60 + "\n")

from train import train
from train_emotion import train_emotion_model
from train_condition import train_condition_model
from train_sentiment import train_sentiment_model

train()
train_emotion_model()
train_condition_model()
train_sentiment_model()

print("=" * 60)
print("  ALL 4 MODELS TRAINED AND SAVED SUCCESSFULLY")
print("=" * 60)
