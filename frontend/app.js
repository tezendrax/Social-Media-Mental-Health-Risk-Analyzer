'use strict';

const SAMPLES = {
    high:    "I can't go on anymore. Everything feels completely hopeless and I don't see any point in continuing. I'm so exhausted from existing.",
    moderate:"I've been feeling really anxious and overwhelmed lately. Work stress is through the roof and I can't sleep properly. Not sure how to cope.",
    low:     "Had an amazing hike with friends today! Feeling so grateful and energized. Life is genuinely good right now.",
    ptsd:    "I keep having flashbacks to what happened. Certain sounds trigger intense panic in me and I've been avoiding anything that reminds me of the incident. I don't feel safe.",
    burnout: "I'm completely exhausted and drained by my work. I used to love what I do but now I dread every morning. I feel like an empty shell just going through the motions.",
};

const SENTIMENT_VALENCE_MAP = {
    "Very Negative": 0.0, "Negative": 0.25, "Neutral": 0.5,
    "Positive": 0.75, "Very Positive": 1.0,
};
const PROB_COLORS = {"High Risk":"#ef4444","Moderate Risk":"#f59e0b","Low Risk":"#10b981"};
const RISK_COLORS = {
    "High Risk":     {stroke:"#ef4444",badge:"#fca5a5",bg:"rgba(239,68,68,.15)",border:"#ef4444"},
    "Moderate Risk": {stroke:"#f59e0b",badge:"#fcd34d",bg:"rgba(245,158,11,.15)",border:"#f59e0b"},
    "Low Risk":      {stroke:"#10b981",badge:"#6ee7b7",bg:"rgba(16,185,129,.15)",border:"#10b981"},
};
const WELLBEING_STATUS = [
    {max:20,label:"Critical",color:"#ef4444"},
    {max:35,label:"Very Low",color:"#f97316"},
    {max:50,label:"Low",color:"#f59e0b"},
    {max:65,label:"Moderate",color:"#a3e635"},
    {max:80,label:"Good",color:"#34d399"},
    {max:100,label:"Excellent",color:"#10b981"},
];
const GAUGE_LEN = 251.2, RING_LEN = 201.06;
const API = 'http://127.0.0.1:8000';
const $ = id => document.getElementById(id);

let selectedPlatform = 'generic', sessionAnalyses = 0;

document.addEventListener('DOMContentLoaded', () => {
    checkModelStatus();
    loadHistory();
    setupListeners();
});

function setupListeners() {
    const textInput = $('textInput');
    textInput.addEventListener('input', () => {
        const l = textInput.value.length;
        $('charCount').textContent = `${l} / 5000`;
        $('charCount').style.color = l > 4500 ? '#ef4444' : '';
    });
    document.querySelectorAll('.platform-btn').forEach(b => {
        b.addEventListener('click', () => {
            document.querySelectorAll('.platform-btn').forEach(x => { x.classList.remove('active'); x.setAttribute('aria-pressed','false'); });
            b.classList.add('active'); b.setAttribute('aria-pressed','true');
            selectedPlatform = b.dataset.platform;
        });
    });
    document.querySelectorAll('.chip').forEach(c => {
        c.addEventListener('click', () => {
            const t = SAMPLES[c.dataset.sample] || '';
            textInput.value = t;
            $('charCount').textContent = `${t.length} / 5000`;
        });
    });
    $('analyzeBtn').addEventListener('click', runAnalysis);
    textInput.addEventListener('keydown', e => { if ((e.ctrlKey||e.metaKey) && e.key==='Enter') runAnalysis(); });
    $('clearHistoryBtn').addEventListener('click', () => { localStorage.removeItem('sujhaav_v3'); renderHistory([]); });
}

async function checkModelStatus() {
    const dot = $('statusDot'), txt = $('modelStatusText');
    try {
        const r = await fetch(`${API}/status`, {signal:AbortSignal.timeout(4000)});
        if (!r.ok) throw new Error();
        const d = await r.json();
        const n = (d.models_available||[]).length;
        dot.className = 'status-dot online';
        txt.textContent = n >= 4 ? 'Full Profile Ready' : n > 0 ? `${n}/4 Models` : 'Heuristic Mode';
    } catch {
        dot.className = 'status-dot error';
        txt.textContent = 'API Offline';
    }
}

async function runAnalysis() {
    const text = $('textInput').value.trim();
    if (!text) { $('textInput').focus(); return; }
    setLoading(true); resetPipeline();
    try {
        await animatePipeline();
        const r = await fetch(`${API}/analyze`, {
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({text, platform: selectedPlatform}),
        });
        if (!r.ok) { const e = await r.json().catch(()=>{}); throw new Error(e?.detail||'API error'); }
        const data = await r.json();
        renderProfile(data);
        addHistory(data);
        sessionAnalyses++;
        $('sessionCount').textContent = sessionAnalyses;
    } catch(err) {
        console.warn('[Sujhaav] offline demo mode:', err.message);
        renderProfile(buildOfflineProfile(text));
        sessionAnalyses++;
        $('sessionCount').textContent = sessionAnalyses;
    } finally {
        setLoading(false);
    }
}

function setLoading(on) {
    const btn = $('analyzeBtn');
    btn.disabled = on;
    btn.querySelector('.btn-inner').classList.toggle('hidden', on);
    btn.querySelector('.btn-loader').classList.toggle('hidden', !on);
}

function delay(ms){ return new Promise(r => setTimeout(r, ms)); }
function resetPipeline(){ document.querySelectorAll('.pipeline-step').forEach(s => s.classList.remove('active','done')); }
async function animatePipeline() {
    const ids = ['step-input','step-preprocess','step-tfidf','step-model','step-output'];
    for(let i=0;i<ids.length;i++){
        if(i>0){ const p=$(ids[i-1]); p.classList.remove('active'); p.classList.add('done'); }
        $(ids[i]).classList.add('active');
        await delay(i<ids.length-1?360:180);
    }
    await delay(200);
    $(ids[ids.length-1]).classList.remove('active'); $(ids[ids.length-1]).classList.add('done');
}

// ── RENDER ────────────────────────────────────────────────────────────────────
function renderProfile(data) {
    const profile = data.profile || {};
    const risk      = profile.risk      || data.prediction || {};
    const emotion   = profile.emotion   || {};
    const condition = profile.condition || {};
    const sentiment = profile.sentiment || {};
    const wellbeing = profile.wellbeing_score ?? 50;

    $('emptyState').classList.add('hidden');
    $('resultsPanel').classList.remove('hidden');

    renderWellbeing(wellbeing);
    renderRiskGauge(risk);
    renderEmotion(emotion);
    renderSentiment(sentiment);
    renderCondition(condition);
    renderProbBars(risk.probabilities || {});
    renderSignals(risk.signals || [], risk);
    renderMeta(data.input_text, risk.model_used);
}

function renderWellbeing(score) {
    const ring = $('wellbeingRing'), val = $('wellbeingValue'), status = $('wellbeingStatus');
    const pct  = Math.min(Math.max(score, 0), 100) / 100;
    const offset = RING_LEN - pct * RING_LEN;
    const ws = WELLBEING_STATUS.find(s => score <= s.max) || WELLBEING_STATUS[WELLBEING_STATUS.length-1];
    requestAnimationFrame(() => {
        ring.style.strokeDashoffset = offset;
        ring.style.stroke = ws.color;
        val.textContent   = score;
        val.style.color   = ws.color;
    });
    status.textContent = ws.label;
    status.style.color = ws.color;
    status.style.borderColor = ws.color;
    status.style.background  = ws.color + '22';
}

function renderRiskGauge(risk) {
    const cls    = risk.classification || 'Low Risk';
    const prob   = risk.probability || 0;
    const conf   = risk.confidence  || 0;
    const colors = RISK_COLORS[cls] || RISK_COLORS['Low Risk'];
    const offset = GAUGE_LEN - Math.min(prob,1) * GAUGE_LEN;
    const angle  = -90 + Math.min(prob,1) * 180;
    requestAnimationFrame(() => {
        $('gaugeFill').style.strokeDashoffset = offset;
        $('gaugeFill').style.stroke = colors.stroke;
        $('gaugeNeedle').style.transform = `rotate(${angle}deg)`;
        $('gaugePct').textContent = `${Math.round(prob*100)}%`;
        $('gaugePct').style.color = colors.stroke;
    });
    const badge = $('classificationBadge');
    badge.textContent = cls;
    badge.style.color = colors.badge;
    badge.style.background = colors.bg;
    badge.style.borderColor = colors.border;
    $('confidenceValue').textContent = `${(conf*100).toFixed(1)}%`;
}

function renderEmotion(emotion) {
    $('emotionIcon').textContent = emotion.icon || '😐';
    $('emotionName').textContent = emotion.dominant || '--';
    const top3 = emotion.top3 || [];
    const bars = $('emotionBars');
    bars.innerHTML = '';
    const maxP = top3.length ? Math.max(...top3.map(e=>e.probability), 0.001) : 1;
    top3.forEach((e,i) => {
        const row = document.createElement('div');
        row.className = 'emo-row';
        row.innerHTML = `<span class="emo-icon">${e.icon}</span>
            <span class="emo-name">${e.emotion}</span>
            <div class="emo-track"><div class="emo-fill" data-target="${(e.probability/maxP*100).toFixed(1)}"></div></div>
            <span class="emo-pct">${(e.probability*100).toFixed(0)}%</span>`;
        bars.appendChild(row);
    });
    requestAnimationFrame(() => {
        bars.querySelectorAll('.emo-fill').forEach((f,i) => {
            setTimeout(() => { f.style.width = `${f.dataset.target}%`; }, i*100);
        });
    });
}

function renderSentiment(sentiment) {
    const label = sentiment.label || 'Neutral';
    const valence = SENTIMENT_VALENCE_MAP[label] ?? 0.5;
    const conf    = sentiment.confidence || 0;
    $('sentimentLabel').textContent = label;
    $('sentimentConf').textContent  = `${(conf*100).toFixed(1)}%`;
    setTimeout(() => {
        $('sentimentMarker').style.left = `${valence*100}%`;
    }, 200);
    // Mini bars
    const bars = $('sentimentBars');
    if(bars) {
        bars.innerHTML = '';
        const dist = sentiment.distribution || {};
        const order = ['Very Negative','Negative','Neutral','Positive','Very Positive'];
        order.forEach(lbl => {
            const v = dist[lbl] || 0;
            const row = document.createElement('div');
            row.className = 'prob-row';
            row.innerHTML = `<div class="prob-row-header"><span class="prob-row-label">${lbl}</span><span class="prob-row-val">${(v*100).toFixed(1)}%</span></div>
                <div class="prob-track"><div class="prob-fill" style="background:#a78bfa" data-t="${(v*100).toFixed(1)}"></div></div>`;
            bars.appendChild(row);
        });
        requestAnimationFrame(() => {
            bars.querySelectorAll('.prob-fill').forEach(f => { f.style.width = `${f.dataset.t}%`; });
        });
    }
}

function renderCondition(condition) {
    const cards = $('conditionCards');
    cards.innerHTML = '';
    const profile = condition.profile || [];
    if(!profile.length) {
        cards.innerHTML = '<p style="font-size:.8rem;color:var(--muted)">No condition data available.</p>';
        return;
    }
    profile.forEach((c,i) => {
        const pct = (c.probability*100).toFixed(1);
        const card = document.createElement('div');
        card.className = 'condition-card';
        card.innerHTML = `
            <span class="cond-rank">#${i+1}</span>
            <span class="cond-color-dot" style="background:${c.color}"></span>
            <span class="cond-name">${c.condition}</span>
            <div class="cond-track"><div class="cond-fill" style="background:${c.color}" data-t="${pct}"></div></div>
            <span class="cond-pct">${pct}%</span>`;
        cards.appendChild(card);
    });
    requestAnimationFrame(() => {
        cards.querySelectorAll('.cond-fill').forEach((f,i) => {
            setTimeout(() => { f.style.width = `${f.dataset.t}%`; }, i*120);
        });
    });
}

function renderProbBars(probs) {
    const el = $('probBars'); el.innerHTML = '';
    ['High Risk','Moderate Risk','Low Risk'].forEach(lbl => {
        const v = probs[lbl] || 0;
        const row = document.createElement('div');
        row.className = 'prob-row';
        row.innerHTML = `<div class="prob-row-header"><span class="prob-row-label">${lbl}</span><span class="prob-row-val">${(v*100).toFixed(1)}%</span></div>
            <div class="prob-track"><div class="prob-fill" style="background:${PROB_COLORS[lbl]}" data-t="${(v*100).toFixed(1)}"></div></div>`;
        el.appendChild(row);
    });
    requestAnimationFrame(() => { el.querySelectorAll('.prob-fill').forEach(f => { f.style.width=`${f.dataset.t}%`; }); });
}

function renderSignals(signals, risk) {
    const bars = $('signalBars');
    $('signalCount').textContent = signals.length;
    bars.innerHTML = '';
    const color = (RISK_COLORS[risk.classification]||RISK_COLORS['Low Risk']).stroke;
    if(!signals.length){ $('signalsEmpty').classList.remove('hidden'); return; }
    $('signalsEmpty').classList.add('hidden');
    const maxC = Math.max(...signals.map(s=>s.contribution), .001);
    signals.forEach((s,i) => {
        const row = document.createElement('div');
        row.className = 'signal-row';
        row.innerHTML = `<span class="signal-kw" title="${s.keyword}">${s.keyword}</span>
            <div class="signal-track"><div class="signal-fill" style="background:${color}" data-t="${(s.contribution/maxC*100).toFixed(1)}"></div></div>
            <span class="signal-val">${s.contribution.toFixed(3)}</span>`;
        bars.appendChild(row);
    });
    bars.querySelectorAll('.signal-fill').forEach((f,i) => {
        setTimeout(() => { f.style.width=`${f.dataset.t}%`; }, i*70);
    });
}

function renderMeta(inputText, modelUsedVal) {
    $('displayInput').textContent = inputText.length > 55 ? `"${inputText.substring(0,55)}..."` : `"${inputText}"`;
    $('modelUsed').textContent    = modelUsedVal || 'unknown';
}

// ── Offline demo ──────────────────────────────────────────────────────────────
function buildOfflineProfile(text) {
    const t = text.toLowerCase();
    const highKw = ['hopeless','suicidal','worthless','depressed','give up','cant go on','dead','empty'];
    const modKw  = ['anxious','stressed','overwhelmed','sad','lonely','exhausted','tired','burnout'];
    let prob = 0.08; const sigs = [];
    highKw.forEach(k => { if(t.includes(k)){ prob+=.2; sigs.push({keyword:k,contribution:.18,risk_level:'high'}); }});
    modKw.forEach(k  => { if(t.includes(k)){ prob+=.08; sigs.push({keyword:k,contribution:.08,risk_level:'moderate'}); }});
    prob = Math.min(prob + Math.random()*.06, .97);
    const cls = prob>.6?'High Risk':prob>.28?'Moderate Risk':'Low Risk';
    const sentLabel = prob>.6?'Very Negative':prob>.4?'Negative':prob>.2?'Neutral':'Positive';
    const emotion   = prob>.6?{dominant:'Hopelessness',icon:'😞',top3:[{emotion:'Hopelessness',probability:.5,icon:'😞'},{emotion:'Sadness',probability:.3,icon:'😢'},{emotion:'Fear',probability:.15,icon:'😨'}]}
                            :prob>.35?{dominant:'Anxiety',icon:'😰',top3:[{emotion:'Anxiety',probability:.45,icon:'😰'},{emotion:'Sadness',probability:.3,icon:'😢'},{emotion:'Fear',probability:.2,icon:'😨'}]}
                            :{dominant:'Joy',icon:'😊',top3:[{emotion:'Joy',probability:.6,icon:'😊'},{emotion:'Neutral',probability:.25,icon:'😐'},{emotion:'Sadness',probability:.1,icon:'😢'}]};
    const condition = prob>.6?{primary:'Depression',color:'#818cf8',profile:[{condition:'Depression',probability:.5,color:'#818cf8'},{condition:'Suicidal Ideation',probability:.3,color:'#ef4444'},{condition:'Loneliness/Isolation',probability:.2,color:'#06b6d4'}]}
                            :prob>.35?{primary:'Anxiety Disorder',color:'#f59e0b',profile:[{condition:'Anxiety Disorder',probability:.45,color:'#f59e0b'},{condition:'Burnout',probability:.3,color:'#fb923c'},{condition:'Depression',probability:.2,color:'#818cf8'}]}
                            :{primary:'Healthy',color:'#10b981',profile:[{condition:'Healthy',probability:.7,color:'#10b981'},{condition:'Burnout',probability:.15,color:'#fb923c'},{condition:'Anxiety Disorder',probability:.1,color:'#f59e0b'}]};
    const wellbeing = Math.max(5, Math.min(95, Math.round(100 - prob*70)));
    return {
        input_text: text, platform: selectedPlatform, timestamp: new Date().toISOString(),
        profile: {
            risk: { classification:cls, probability:parseFloat(prob.toFixed(3)), confidence:parseFloat((.65+Math.random()*.2).toFixed(3)),
                probabilities:{'High Risk':parseFloat(prob.toFixed(3)),'Moderate Risk':parseFloat(Math.min(1-prob,.45).toFixed(3)),'Low Risk':parseFloat(Math.max(1-prob-.15,.03).toFixed(3))},
                signals:sigs.slice(0,6), model_used:'offline_demo'},
            emotion, condition,
            sentiment:{label:sentLabel,valence:{VeryNegative:-1,Negative:-.5,Neutral:0,Positive:.5,VeryPositive:1}[sentLabel]||0,confidence:.72,distribution:{}},
            wellbeing_score: wellbeing,
        }
    };
}

// ── History ───────────────────────────────────────────────────────────────────
function addHistory(data) {
    const p = data.profile||{}, r = p.risk||data.prediction||{};
    const entry = {
        timestamp:   data.timestamp||new Date().toISOString(),
        platform:    data.platform||'generic',
        text_preview:(data.input_text||'').substring(0,55)+(data.input_text?.length>55?'…':''),
        classification: r.classification,
        probability:    r.probability,
        confidence:     r.confidence,
        emotion:        p.emotion?.dominant||'—',
        condition:      p.condition?.primary||'—',
        sentiment:      p.sentiment?.label||'—',
        wellbeing_score:p.wellbeing_score??50,
        model_used:     r.model_used||'unknown',
    };
    const stored = getHistory();
    stored.unshift(entry);
    if(stored.length>10) stored.pop();
    localStorage.setItem('sujhaav_v3', JSON.stringify(stored));
    renderHistory(stored);
}
function getHistory(){ try{ return JSON.parse(localStorage.getItem('sujhaav_v3')||'[]'); }catch{ return []; } }
function loadHistory(){ renderHistory(getHistory()); }
function renderHistory(items) {
    const empty=$('historyEmpty'), table=$('historyTable'), body=$('historyBody');
    if(!items||!items.length){ empty.classList.remove('hidden'); table.classList.add('hidden'); return; }
    empty.classList.add('hidden'); table.classList.remove('hidden');
    body.innerHTML = items.map(it => {
        const cls = it.classification||'Low Risk';
        const bc  = cls==='High Risk'?'badge-high':cls==='Moderate Risk'?'badge-mod':'badge-low';
        return `<tr>
            <td class="hist-time">${fmtTime(it.timestamp)}</td>
            <td>${esc(it.platform||'generic')}</td>
            <td class="hist-preview" title="${esc(it.text_preview)}">${esc(it.text_preview)}</td>
            <td><span class="hist-badge ${bc}">${esc(cls)}</span></td>
            <td>${esc(it.emotion||'—')}</td>
            <td>${esc(it.condition||'—')}</td>
            <td class="hist-wb">${it.wellbeing_score??'—'}</td>
        </tr>`;
    }).join('');
}
function fmtTime(iso){ try{ return new Date(iso).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'}); }catch{ return '—'; } }
function esc(s){ return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
