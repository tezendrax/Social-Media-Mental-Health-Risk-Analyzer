"""
api/index.py — Vercel Serverless Entry Point for the FastAPI app.
Vercel's Python runtime will look for an 'app' ASGI object in this file.
"""
import sys, os

# Make ml_pipeline/src importable within the serverless environment
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'backend'))
sys.path.insert(0, os.path.join(ROOT, 'ml_pipeline', 'src'))

from app import app  # re-export the FastAPI app for Vercel
