"""
app.py — FastAPI backend (v3 — Full Mental Health Profile)
Endpoints:
    GET  /           health check
    GET  /status     model status
    POST /analyze    full profile (risk + emotion + condition + sentiment + wellbeing)
    POST /signals    keyword signals only
    GET  /history    last 20 analyses
    GET  /stats      session aggregate stats
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from collections import deque
import statistics

from ml_service import ml_service

app = FastAPI(title="Mental Health Risk Analyzer API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

_history: deque = deque(maxlen=20)


class AnalyzeRequest(BaseModel):
    text:     str = Field(..., min_length=3, max_length=5000)
    platform: Optional[str] = Field("generic")


class SignalsRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=5000)


@app.get("/", tags=["Health"])
def health_check():
    return {"status": "healthy", "version": "3.0.0",
            "message": "Mental Health Profiling API is running."}


@app.get("/status", tags=["Health"])
def model_status():
    return ml_service.get_status()


@app.post("/analyze", tags=["Analysis"])
def analyze_text(request: AnalyzeRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    profile = ml_service.profile_text(request.text)
    result  = {
        "input_text": request.text,
        "platform":   request.platform,
        "timestamp":  datetime.utcnow().isoformat() + "Z",
        "profile":    profile,
        # Keep legacy 'prediction' key for backwards compat
        "prediction": profile["risk"],
    }
    _history.append({
        "timestamp":        result["timestamp"],
        "platform":         request.platform,
        "text_preview":     request.text[:60] + ("..." if len(request.text) > 60 else ""),
        "classification":   profile["risk"]["classification"],
        "probability":      profile["risk"]["probability"],
        "confidence":       profile["risk"]["confidence"],
        "emotion":          profile["emotion"]["dominant"],
        "condition":        profile["condition"]["primary"],
        "sentiment":        profile["sentiment"]["label"],
        "wellbeing_score":  profile["wellbeing_score"],
        "model_used":       profile["risk"].get("model_used", "unknown"),
    })
    return result


@app.post("/signals", tags=["Analysis"])
def get_signals(request: SignalsRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    return {"signals": ml_service.get_signals_for_text(request.text)}


@app.get("/history", tags=["History"])
def get_history(limit: int = 10):
    limit = min(limit, 20)
    items = list(_history)[-limit:]
    items.reverse()
    return {"history": items, "total": len(_history)}


@app.get("/stats", tags=["History"])
def get_stats():
    if not _history:
        return {"total_analyses": 0, "message": "No analyses yet."}
    items = list(_history)
    total = len(items)
    clss  = [i["classification"] for i in items]
    probs = [i["probability"]    for i in items]
    confs = [i["confidence"]     for i in items]
    wbs   = [i.get("wellbeing_score", 50) for i in items]
    return {
        "total_analyses":        total,
        "classification_counts": {
            "High Risk":     clss.count("High Risk"),
            "Moderate Risk": clss.count("Moderate Risk"),
            "Low Risk":      clss.count("Low Risk"),
        },
        "avg_risk_probability":  round(statistics.mean(probs), 3),
        "avg_confidence":        round(statistics.mean(confs), 3),
        "avg_wellbeing_score":   round(statistics.mean(wbs),   1),
        "high_risk_percentage":  round(clss.count("High Risk") / total * 100, 1),
        "model_used":            items[-1].get("model_used", "unknown"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
