"""
UNSW-NB15 Network Intrusion Detection System
Streamlit Demo App
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json
import os
import joblib
import time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Network IDS — UNSW-NB15",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
}
.stApp { background: #0a0e1a; color: #e0e6f0; }

/* Header */
.ids-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #112240 50%, #0d1b2a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.ids-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #00d4ff, #0080ff, #7b2fff, #00d4ff);
    background-size: 300% 100%;
    animation: shimmer 3s linear infinite;
}
@keyframes shimmer { 0%{background-position:0%} 100%{background-position:300%} }

.ids-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2rem;
    color: #00d4ff;
    margin: 0;
    letter-spacing: 2px;
}
.ids-subtitle {
    color: #8899bb;
    font-size: 0.95rem;
    margin-top: 0.4rem;
    font-weight: 300;
}

/* Metric cards */
.metric-card {
    background: #0f1829;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: #00d4ff; }
.metric-val { font-size: 2rem; font-weight: 800; color: #00d4ff; }
.metric-label { font-size: 0.8rem; color: #8899bb; text-transform: uppercase; letter-spacing: 1px; }

/* Attack badge */
.attack-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.1rem;
    font-weight: bold;
    letter-spacing: 1px;
}
.badge-normal   { background:#0d3320; color:#00e676; border:1px solid #00e676; }
.badge-attack   { background:#2d0f0f; color:#ff5252; border:1px solid #ff5252; }
.badge-unknown  { background:#1a1a2e; color:#ffd740; border:1px solid #ffd740; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1b2a !important;
    border-right: 1px solid #1e3a5f;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0080ff, #00d4ff);
    color: #0a0e1a;
    font-family: 'Share Tech Mono', monospace;
    font-weight: bold;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    letter-spacing: 1px;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Terminal log box */
.terminal {
    background: #050810;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    color: #00d4ff;
    line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_DIR   = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
ATTACK_CATS = ['analysis', 'backdoor', 'dos', 'exploits', 'fuzzers',
               'generic', 'normal', 'reconnaissance', 'shellcode', 'worms']

ATTACK_COLORS = {
    'normal':        '#00e676',
    'dos':           '#ff5252',
    'exploits':      '#ff6d00',
    'generic':       '#ffd740',
    'reconnaissance':'#40c4ff',
    'fuzzers':       '#e040fb',
    'backdoor':      '#ff4081',
    'shellcode':     '#ff6e40',
    'worms':         '#b2ff59',
    'analysis':      '#80d8ff',
}

ATTACK_DESCRIPTIONS = {
    'normal':        '✅ Legitimate network traffic. No threat detected.',
    'dos':           '🔴 Denial of Service — Flood attack to exhaust resources.',
    'exploits':      '🟠 Exploit — Leverages known software vulnerabilities.',
    'generic':       '🟡 Generic Attack — Signature-based attack pattern.',
    'reconnaissance':'🔵 Reconnaissance — Network scanning / info gathering.',
    'fuzzers':       '🟣 Fuzzer — Random/malformed input injection.',
    'backdoor':      '🔴 Backdoor — Covert persistent access attempt.',
    'shellcode':     '🟠 Shellcode — Payload delivering shell execution.',
    'worms':         '🟢 Worm — Self-propagating malicious code.',
    'analysis':      '🔵 Analysis — Packet inspection / port scanning.',
}

# ── Load Models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf = None
    svm = None
    mlp = None
    scaler = None
    selector = None
    le = None
    meta = {}

    try:
        scaler   = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        selector = joblib.load(os.path.join(MODEL_DIR, 'feature_selector.pkl'))
        le       = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    except Exception as e:
        st.error(f"Error loading preprocessing objects (scaler/selector/encoder): {e}")
        return None, None, None, None, None, None, {}, False

    try:
        if os.path.exists(os.path.join(MODEL_DIR, 'metadata.json')):
            with open(os.path.join(MODEL_DIR, 'metadata.json')) as f:
                meta = json.load(f)
    except Exception as e:
        pass

    try:
        rf = joblib.load(os.path.join(MODEL_DIR, 'random_forest.pkl'))
    except Exception as e:
        pass

    try:
        svm = joblib.load(os.path.join(MODEL_DIR, 'svm.pkl'))
    except Exception as e:
        pass

    try:
        import tensorflow as tf
        mlp = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'mlp_model.keras'))
    except Exception as e:
        pass

    # App is considered loaded if preprocessing is available and at least one model is present
    models_loaded = (scaler is not None and selector is not None and le is not None and 
                     (rf is not None or svm is not None or mlp is not None))

    return rf, svm, mlp, scaler, selector, le, meta, models_loaded

rf_model, svm_model, mlp_model, scaler, selector, le_target, metadata, models_loaded = load_models()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ IDS Control Panel")
    st.markdown("---")

    page = st.radio("Navigate", [
        "🏠  Overview",
        "🔬  Live Prediction",
        "📊  Model Performance",
        "📖  About Dataset",
    ])

    st.markdown("---")
    st.markdown("**Model Status**")
    if models_loaded:
        st.success("Models loaded ✓")
        loaded_list = []
        if rf_model is not None: loaded_list.append("Random Forest")
        if svm_model is not None: loaded_list.append("SVM")
        if mlp_model is not None: loaded_list.append("MLP")
        st.caption(f"Available: {', '.join(loaded_list)}")
        
        if metadata.get('results'):
            r = metadata['results']
            mlp_f1_val = r.get('mlp', {}).get('f1')
            if mlp_f1_val:
                st.metric("Best F1 (MLP)", f"{mlp_f1_val:.3f}")
    else:
        st.warning("⚠️ Models not found.\nRun the Colab notebook first, then copy `saved_models/` here.")

    st.markdown("---")
    available_options = []
    if mlp_model is not None:
        available_options.append("MLP (Deep Learning)")
    if rf_model is not None:
        available_options.append("Random Forest")
    if svm_model is not None:
        available_options.append("SVM")
    
    if not available_options:
        available_options = ["No models loaded"]
        
    selected_model = st.selectbox("Active Model", available_options)
    st.markdown("---")
    st.caption("UNSW-NB15 Dataset · ACCS © 2015")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ids-header">
    <p class="ids-title">⟨ NETWORK INTRUSION DETECTION SYSTEM ⟩</p>
    <p class="ids-subtitle">UNSW-NB15 Dataset · Random Forest · SVM · MLP Deep Learning · Multiclass Classification</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    col1, col2, col3, col4 = st.columns(4)

    res = metadata.get('results', {})
    rf_acc  = res.get('random_forest', {}).get('accuracy', 0)
    svm_acc = res.get('svm',           {}).get('accuracy', 0)
    mlp_acc = res.get('mlp',           {}).get('accuracy', 0)
    mlp_f1  = res.get('mlp',           {}).get('f1', 0)

    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{rf_acc:.1%}</div>
            <div class="metric-label">Random Forest Acc</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{svm_acc:.1%}</div>
            <div class="metric-label">SVM Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{mlp_acc:.1%}</div>
            <div class="metric-label">MLP Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{mlp_f1:.3f}</div>
            <div class="metric-label">MLP Weighted F1</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1.2, 1])

    with col_l:
        st.markdown("#### 🔄 Pipeline Architecture")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_facecolor('#0f1829')
        ax.set_facecolor('#0f1829')
        ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis('off')

        steps = [
            (0.5, "Raw\nTraffic", "#1e3a5f"),
            (2.2, "Pre-\nprocess", "#1a3a4a"),
            (3.9, "Feature\nSelect", "#1a2a4a"),
            (5.6, "SMOTE\nBalance", "#1a2a3a"),
            (7.3, "Train\nModels", "#1a1a3a"),
            (9.0, "Predict\nClass", "#0d3320"),
        ]
        for x, label, color in steps:
            rect = mpatches.FancyBboxPatch((x - 0.55, 0.8), 1.1, 1.4,
                boxstyle="round,pad=0.1", facecolor=color, edgecolor='#00d4ff', linewidth=1.2)
            ax.add_patch(rect)
            ax.text(x, 1.5, label, ha='center', va='center', color='#e0e6f0',
                    fontsize=8.5, fontweight='bold', fontfamily='monospace')
            if x < 9.0:
                ax.annotate('', xy=(x + 0.65, 1.5), xytext=(x + 0.55, 1.5),
                            arrowprops=dict(arrowstyle='->', color='#00d4ff', lw=1.5))
        st.pyplot(fig, use_container_width=True)

    with col_r:
        st.markdown("#### 📡 Attack Categories")
        for cat, color in ATTACK_COLORS.items():
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
                <div style="width:12px;height:12px;border-radius:50%;background:{color};flex-shrink:0;"></div>
                <span style="font-family:monospace;font-size:0.85rem;color:#c0cfe0;text-transform:uppercase;">{cat}</span>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif "Prediction" in page:
    st.markdown("#### 🔬 Live Traffic Classification")
    st.info("Enter network flow features below, or use a preset scenario to simulate a packet.")

    col_preset, _ = st.columns([2, 3])
    with col_preset:
        preset = st.selectbox("Load Preset Scenario", [
            "— manual entry —",
            "Normal HTTP Traffic",
            "DoS SYN Flood",
            "Port Scan (Reconnaissance)",
            "Exploit Attempt",
            "Backdoor Connection",
        ])

    # Preset values
    PRESETS = {
        "Normal HTTP Traffic":       dict(dur=0.12,  proto='tcp',  spkts=8,  dpkts=6,  sbytes=1200, dbytes=4800, rate=85.0,  sttl=64,  dttl=128, sload=9600.0,  dload=38400.0, sinpkt=15.0, dinpkt=20.0, sjit=2.1,  djit=1.8,  swin=65535, dwin=65535, stcpb=120000, dtcpb=130000, tcprtt=0.03, synack=0.01, ackdat=0.01, smean=150, dmean=800, trans_depth=1, res_bdy_len=4096, ct_srv_src=3, ct_state_ttl=2, ct_dst_ltm=2, ct_src_dport_ltm=1, ct_dst_sport_ltm=1, ct_dst_src_ltm=3),
        "DoS SYN Flood":             dict(dur=0.001, proto='tcp',  spkts=200,dpkts=0,   sbytes=12000,dbytes=0,    rate=200000,sttl=64,  dttl=0,   sload=96000000,dload=0.0,     sinpkt=0.005,dinpkt=0.0,  sjit=0.1,  djit=0.0,  swin=1024,  dwin=0,     stcpb=0,      dtcpb=0,      tcprtt=0.0,  synack=0.0,  ackdat=0.0,  smean=60,  dmean=0,   trans_depth=0, res_bdy_len=0,    ct_srv_src=200,ct_state_ttl=1,ct_dst_ltm=1,  ct_src_dport_ltm=1,   ct_dst_sport_ltm=1,   ct_dst_src_ltm=200),
        "Port Scan (Reconnaissance)":dict(dur=0.005, proto='tcp',  spkts=1,  dpkts=0,   sbytes=60,   dbytes=0,    rate=200.0, sttl=64,  dttl=0,   sload=96000.0, dload=0.0,     sinpkt=5.0,  dinpkt=0.0,  sjit=0.5,  djit=0.0,  swin=512,   dwin=0,     stcpb=1000,   dtcpb=0,      tcprtt=0.0,  synack=0.0,  ackdat=0.0,  smean=60,  dmean=0,   trans_depth=0, res_bdy_len=0,    ct_srv_src=50, ct_state_ttl=1,ct_dst_ltm=50, ct_src_dport_ltm=50,  ct_dst_sport_ltm=1,   ct_dst_src_ltm=50),
        "Exploit Attempt":           dict(dur=0.45,  proto='tcp',  spkts=15, dpkts=12,  sbytes=8000, dbytes=2000, rate=60.0,  sttl=64,  dttl=128, sload=142222.0,dload=35556.0, sinpkt=30.0, dinpkt=37.5, sjit=5.0,  djit=4.2,  swin=65535, dwin=32768, stcpb=500000, dtcpb=600000, tcprtt=0.08, synack=0.02, ackdat=0.02, smean=533, dmean=167, trans_depth=2, res_bdy_len=512,  ct_srv_src=4,  ct_state_ttl=2,ct_dst_ltm=3,  ct_src_dport_ltm=2,   ct_dst_sport_ltm=2,   ct_dst_src_ltm=4),
        "Backdoor Connection":       dict(dur=300.0, proto='tcp',  spkts=50, dpkts=48,  sbytes=3000, dbytes=2800, rate=0.33,  sttl=64,  dttl=128, sload=80.0,    dload=74.7,    sinpkt=6000.0,dinpkt=6250.0,sjit=200.0,djit=210.0,swin=65535, dwin=65535, stcpb=700000, dtcpb=710000, tcprtt=0.15, synack=0.05, ackdat=0.05, smean=60,  dmean=58,  trans_depth=0, res_bdy_len=0,    ct_srv_src=1,  ct_state_ttl=3,ct_dst_ltm=1,  ct_src_dport_ltm=1,   ct_dst_sport_ltm=1,   ct_dst_src_ltm=1),
    }

    pv = PRESETS.get(preset, PRESETS["Normal HTTP Traffic"]) if preset != "— manual entry —" else {}

    def g(k, default): return pv.get(k, default)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**⏱ Flow Timing**")
        dur      = st.number_input("Duration (sec)",   value=float(g('dur', 0.1)),   step=0.01, format="%.4f")
        rate     = st.number_input("Rate (pkt/s)",     value=float(g('rate', 50.0)), step=1.0)
        sinpkt   = st.number_input("Src Inter-Pkt (ms)",value=float(g('sinpkt',20.0)),step=1.0)
        dinpkt   = st.number_input("Dst Inter-Pkt (ms)",value=float(g('dinpkt',20.0)),step=1.0)
        tcprtt   = st.number_input("TCP RTT",          value=float(g('tcprtt',0.05)), step=0.01,format="%.4f")
        synack   = st.number_input("SYN-ACK time",     value=float(g('synack',0.01)), step=0.001,format="%.4f")
        ackdat   = st.number_input("ACK-DAT time",     value=float(g('ackdat',0.01)), step=0.001,format="%.4f")

    with c2:
        st.markdown("**📦 Packet Counts & Bytes**")
        spkts    = st.number_input("Src Packets",   value=int(g('spkts',8)),    step=1)
        dpkts    = st.number_input("Dst Packets",   value=int(g('dpkts',6)),    step=1)
        sbytes   = st.number_input("Src Bytes",     value=int(g('sbytes',1200)),step=1)
        dbytes   = st.number_input("Dst Bytes",     value=int(g('dbytes',4800)),step=1)
        sload    = st.number_input("Src Load (b/s)",value=float(g('sload',9600.0)),step=100.0)
        dload    = st.number_input("Dst Load (b/s)",value=float(g('dload',38400.0)),step=100.0)
        smean    = st.number_input("Src Mean Pkt",  value=int(g('smean',150)),  step=1)
        dmean    = st.number_input("Dst Mean Pkt",  value=int(g('dmean',800)),  step=1)

    with c3:
        st.markdown("**🌐 Protocol & Connection**")
        proto    = st.selectbox("Protocol", ['tcp','udp','icmp','arp','ospf','other'],
                                index=['tcp','udp','icmp','arp','ospf','other'].index(g('proto','tcp')))
        sttl     = st.number_input("Src TTL",     value=int(g('sttl',64)),   step=1)
        dttl     = st.number_input("Dst TTL",     value=int(g('dttl',128)),  step=1)
        swin     = st.number_input("Src Win Size",value=int(g('swin',65535)),step=1)
        dwin     = st.number_input("Dst Win Size",value=int(g('dwin',65535)),step=1)
        sjit     = st.number_input("Src Jitter",  value=float(g('sjit',2.1)),step=0.1, format="%.2f")
        djit     = st.number_input("Dst Jitter",  value=float(g('djit',1.8)),step=0.1, format="%.2f")

        st.markdown("**🔗 CT Features**")
        ct_srv_src        = st.number_input("ct_srv_src",        value=int(g('ct_srv_src',3)),       step=1)
        ct_state_ttl      = st.number_input("ct_state_ttl",      value=int(g('ct_state_ttl',2)),     step=1)
        ct_dst_ltm        = st.number_input("ct_dst_ltm",        value=int(g('ct_dst_ltm',2)),       step=1)
        ct_src_dport_ltm  = st.number_input("ct_src_dport_ltm",  value=int(g('ct_src_dport_ltm',1)), step=1)
        ct_dst_sport_ltm  = st.number_input("ct_dst_sport_ltm",  value=int(g('ct_dst_sport_ltm',1)), step=1)
        ct_dst_src_ltm    = st.number_input("ct_dst_src_ltm",    value=int(g('ct_dst_src_ltm',3)),   step=1)

    st.markdown("---")
    predict_btn = st.button("🔍  ANALYSE TRAFFIC", use_container_width=False)

    if predict_btn:
        with st.spinner("Analysing packet..."):
            time.sleep(0.6)

        if not models_loaded:
            st.error("⚠️ Models not loaded. Please run the Colab notebook and place `saved_models/` in the project root.")
        else:
            # Build feature vector
            proto_map = {'tcp':0,'udp':1,'icmp':2,'arp':3,'ospf':4,'other':5}
            feature_vector = np.array([[
                dur, proto_map.get(proto,0), spkts, dpkts, sbytes, dbytes, rate,
                sttl, dttl, sload, dload, sinpkt, dinpkt, sjit, djit, swin, dwin,
                g('stcpb',0), g('dtcpb',0), tcprtt, synack, ackdat, smean, dmean,
                g('trans_depth',0), g('res_bdy_len',0), ct_srv_src, ct_state_ttl,
                ct_dst_ltm, ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm,
                # Pad remaining features with zeros up to original feature count
                *([0.0] * 15)
            ]], dtype=np.float32)

            # Truncate or pad to match scaler's expected input
            n_expected = scaler.n_features_in_
            if feature_vector.shape[1] > n_expected:
                feature_vector = feature_vector[:, :n_expected]
            elif feature_vector.shape[1] < n_expected:
                padding = np.zeros((1, n_expected - feature_vector.shape[1]))
                feature_vector = np.hstack([feature_vector, padding])

            try:
                fv_scaled   = scaler.transform(feature_vector)
                fv_selected = selector.transform(fv_scaled)

                model_map = {}
                if mlp_model is not None:
                    model_map["MLP (Deep Learning)"] = ("mlp", mlp_model)
                if rf_model is not None:
                    model_map["Random Forest"] = ("rf",  rf_model)
                if svm_model is not None:
                    model_map["SVM"] = ("svm", svm_model)

                if selected_model not in model_map:
                    st.error(f"Selected model '{selected_model}' is not available.")
                else:
                    mkey, mobj = model_map[selected_model]

                    if mkey == 'mlp':
                        import tensorflow as tf
                        proba = mobj.predict(fv_selected, verbose=0)[0]
                        pred_idx = np.argmax(proba)
                        confidence = float(proba[pred_idx])
                        pred_label = le_target.inverse_transform([pred_idx])[0]
                    else:
                        pred_idx = mobj.predict(fv_selected)[0]
                        pred_label = le_target.inverse_transform([pred_idx])[0]
                        if hasattr(mobj, 'predict_proba'):
                            proba = mobj.predict_proba(fv_selected)[0]
                            confidence = float(proba[pred_idx])
                        else:
                            proba = np.zeros(len(le_target.classes_))
                            proba[pred_idx] = 1.0
                            confidence = 1.0

                is_normal = pred_label == 'normal'
                badge_cls = 'badge-normal' if is_normal else 'badge-attack'

                st.markdown(f"""
                <div style="background:#0f1829;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem 2rem;margin-top:1rem;">
                    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;">
                        <div>
                            <div style="color:#8899bb;font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.4rem;">Classification Result</div>
                            <span class="attack-badge {badge_cls}">{pred_label.upper()}</span>
                        </div>
                        <div style="text-align:right;">
                            <div style="color:#8899bb;font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;">Confidence</div>
                            <div style="font-size:1.8rem;font-weight:800;color:{'#00e676' if is_normal else '#ff5252'};">{confidence:.1%}</div>
                        </div>
                    </div>
                    <div style="margin-top:1rem;color:#a0b4cc;font-size:0.9rem;">{ATTACK_DESCRIPTIONS.get(pred_label,'Unknown attack type.')}</div>
                </div>
                """, unsafe_allow_html=True)

                # Probability bar chart (MLP)
                if mkey == 'mlp' and len(proba) == len(le_target.classes_):
                    st.markdown("<br>**Class Probability Distribution**", unsafe_allow_html=True)
                    labels = le_target.classes_
                    colors_list = [ATTACK_COLORS.get(l, '#888') for l in labels]
                    fig, ax = plt.subplots(figsize=(9, 2.8))
                    fig.patch.set_facecolor('#0f1829')
                    ax.set_facecolor('#0f1829')
                    bars = ax.barh(labels, proba, color=colors_list, edgecolor='none', height=0.6)
                    ax.set_xlim(0, 1)
                    ax.set_xlabel('Probability', color='#8899bb')
                    ax.tick_params(colors='#c0cfe0', labelsize=9)
                    ax.spines[:].set_color('#1e3a5f')
                    for bar, val in zip(bars, proba):
                        if val > 0.02:
                            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                                    f'{val:.3f}', va='center', color='#e0e6f0', fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)

                # Log
                st.markdown(f"""
                <div class="terminal">
                    [SYS]  Model      : {selected_model}<br>
                    [SYS]  Protocol   : {proto.upper()}<br>
                    [SYS]  Duration   : {dur:.4f}s<br>
                    [SYS]  Src Pkts   : {spkts}  |  Dst Pkts: {dpkts}<br>
                    [SYS]  Prediction : <span style="color:{'#00e676' if is_normal else '#ff5252'};font-weight:bold;">{pred_label.upper()}</span><br>
                    [SYS]  Confidence : {confidence:.4f}<br>
                    [SYS]  Status     : {'CLEAR' if is_normal else '⚠ THREAT DETECTED'}
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.caption("Make sure saved_models/ folder is present with all model files.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif "Performance" in page:
    st.markdown("#### 📊 Model Performance Summary")

    res = metadata.get('results', {})
    if not res:
        st.warning("No results found. Run the Colab notebook first.")
    else:
        models_list  = ['Random Forest', 'SVM', 'MLP (Deep Learning)']
        keys         = ['random_forest', 'svm', 'mlp']
        accs         = [res.get(k, {}).get('accuracy', 0) for k in keys]
        f1s          = [res.get(k, {}).get('f1', 0) for k in keys]

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            fig.patch.set_facecolor('#0f1829')
            ax.set_facecolor('#0f1829')
            x = np.arange(3)
            bars = ax.bar(x, accs, color=['#2196F3','#FF5722','#4CAF50'], width=0.5, edgecolor='none')
            ax.set_xticks(x); ax.set_xticklabels(models_list, color='#c0cfe0', fontsize=9)
            ax.set_ylabel('Accuracy', color='#8899bb')
            ax.set_ylim(0, 1.1)
            ax.tick_params(colors='#8899bb')
            ax.spines[:].set_color('#1e3a5f')
            ax.set_title('Accuracy Comparison', color='#e0e6f0', fontweight='bold')
            for bar, v in zip(bars, accs):
                ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f'{v:.3f}',
                        ha='center', color='#e0e6f0', fontsize=9, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with col2:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            fig.patch.set_facecolor('#0f1829')
            ax.set_facecolor('#0f1829')
            bars = ax.bar(x, f1s, color=['#2196F3','#FF5722','#4CAF50'], width=0.5, edgecolor='none')
            ax.set_xticks(x); ax.set_xticklabels(models_list, color='#c0cfe0', fontsize=9)
            ax.set_ylabel('Weighted F1-Score', color='#8899bb')
            ax.set_ylim(0, 1.1)
            ax.tick_params(colors='#8899bb')
            ax.spines[:].set_color('#1e3a5f')
            ax.set_title('F1-Score Comparison', color='#e0e6f0', fontweight='bold')
            for bar, v in zip(bars, f1s):
                ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f'{v:.3f}',
                        ha='center', color='#e0e6f0', fontsize=9, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        st.markdown("#### 📋 Results Table")
        df_res = pd.DataFrame({
            'Model': models_list,
            'Accuracy': [f"{v:.4f}" for v in accs],
            'F1-Score': [f"{v:.4f}" for v in f1s],
        })
        st.dataframe(df_res, use_container_width=True, hide_index=True)

        st.markdown("#### 📈 Training & Evaluation Visualizations")
        col_v1, col_v2 = st.columns(2)
        visuals_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'visuals'))

        with col_v1:
            cm_path = os.path.join(visuals_dir, 'confusion_matrices.png')
            if os.path.exists(cm_path):
                st.image(cm_path, caption='Confusion Matrices for trained models', use_container_width=True)

        with col_v2:
            curves_path = os.path.join(visuals_dir, 'mlp_training_curves.png')
            if os.path.exists(curves_path):
                st.image(curves_path, caption='MLP Training Loss & Accuracy Curves', use_container_width=True)

        col_v3, col_v4 = st.columns(2)
        with col_v3:
            fi_path = os.path.join(visuals_dir, 'feature_importance.png')
            if os.path.exists(fi_path):
                st.image(fi_path, caption='Random Forest Feature Importance', use_container_width=True)
        with col_v4:
            dist_path = os.path.join(visuals_dir, 'class_distribution.png')
            if os.path.exists(dist_path):
                st.image(dist_path, caption='UNSW-NB15 Class Distribution', use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif "Dataset" in page:
    st.markdown("#### 📖 UNSW-NB15 Dataset")

    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown("""
        The **UNSW-NB15** dataset was developed by the **Australian Centre for Cyber Security (ACCS)**
        using the IXIA PerfectStorm tool to simulate realistic network traffic.

        | Property | Value |
        |---|---|
        | Total Records | 2,540,044 |
        | Training Set | 175,341 |
        | Testing Set | 82,332 |
        | Features | 49 |
        | Attack Categories | 9 + Normal |
        | Year | 2015 |

        **Feature categories:** Basic, Content, Time, Generated, and Label features.

        **Citation:** Moustafa & Slay (2015). *UNSW-NB15: A Comprehensive Data Set for
        Network Intrusion Detection Systems.* MilCIS, IEEE.
        """)
    with col2:
        # Attack category pie chart
        approx_dist = {
            'generic': 40000, 'exploits': 33393, 'fuzzers': 18184,
            'dos': 12264, 'reconnaissance': 10491, 'analysis': 2000,
            'backdoor': 1746, 'shellcode': 1133, 'worms': 130, 'normal': 56000,
        }
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor('#0f1829')
        ax.set_facecolor('#0f1829')
        colors_list = [ATTACK_COLORS[k] for k in approx_dist]
        wedges, texts, autotexts = ax.pie(
            approx_dist.values(), labels=approx_dist.keys(),
            autopct='%1.1f%%', colors=colors_list,
            textprops={'color':'#c0cfe0','fontsize':8},
            pctdistance=0.8, startangle=140
        )
        for at in autotexts: at.set_fontsize(7)
        ax.set_title('Approx. Class Distribution\n(Training Set)', color='#e0e6f0', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
