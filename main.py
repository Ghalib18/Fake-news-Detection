import streamlit as st
import joblib
import numpy as np
from datetime import datetime


# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Fake News Detection (Majority Voting)",
    page_icon="📰",
    layout="wide"
)

# ------------------ CSS (Nice UI) ------------------
st.markdown("""
<style>
    .main{background-color:#0e1117;}
    .block-container{padding-top:1.5rem;}
    .titlebox{
        padding:18px 22px;
        border-radius:18px;
        background: linear-gradient(135deg, #111827, #0b1220);
        border:1px solid rgba(255,255,255,0.10);
    }
    .card{
        padding:16px 18px;
        border-radius:16px;
        border:1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.04);
    }
    .badge-real{
        display:inline-block;
        padding:6px 14px;
        border-radius:999px;
        font-weight:800;
        color:#0b1220;
        background:#22c55e;
    }
    .badge-fake{
        display:inline-block;
        padding:6px 14px;
        border-radius:999px;
        font-weight:800;
        color:#0b1220;
        background:#ef4444;
    }
    .muted{opacity:0.75;font-size:13px;}
</style>
""", unsafe_allow_html=True)


# ------------------ Load Models ------------------
@st.cache_resource
def load_artifacts():
    tfidf = joblib.load("tfidf.pkl")

    # ✅ Model names changed only for UI display
    models = {
        "Logistic Regression": joblib.load("lr.pkl"),
        "Naive Bayes": joblib.load("Decision.pkl"),      # was Decision Tree
        "SVM": joblib.load("random.pkl"),                # was Random Forest
        "Model 2": joblib.load("Boosting.pkl"),          # was Boosting
    }
    return tfidf, models


# ------------------ Helper functions ------------------
def get_prediction(model, X):
    """Return prediction (0/1) and confidence (if available)."""
    pred = model.predict(X)[0]

    conf = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        conf = float(np.max(proba))

    return int(pred), conf


def majority_vote(pred_dict):
    """Majority vote from predictions. Tie -> FAKE by default."""
    preds = list(pred_dict.values())
    ones = preds.count(1)
    zeros = preds.count(0)
    final = 1 if ones > zeros else 0
    return final, ones, zeros


# ------------------ UI Header ------------------
st.markdown("""
<div class="titlebox">
    <h1 style="margin:0;">📰 Fake News Detection System</h1>
    <p style="margin:6px 0 0 0;" class="muted">
        Majority Voting Ensemble (LR + Naive Bayes + SVM + Model 2)
    </p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ------------------ Load artifacts safely ------------------
try:
    tfidf, models = load_artifacts()
except Exception as e:
    st.error("❌ Error loading models/vectorizer.")
    st.code(str(e))
    st.stop()


# ------------------ Layout ------------------
left, right = st.columns([1.3, 1])

with left:
    st.subheader("✍️ Paste News Text")
    news_text = st.text_area(
        "Enter full news article/content",
        height=280,
        placeholder="Paste full news content here..."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        detect_btn = st.button("🔍 Detect News", use_container_width=True)
    with col2:
        clear_btn = st.button("🧹 Clear", use_container_width=True)

    if clear_btn:
        st.session_state.clear()
        st.rerun()


with right:
    st.subheader("📌 Final Output")

    if detect_btn:
        if not news_text.strip():
            st.warning("⚠️ Please paste some news text.")
        else:
            # Transform text
            X = tfidf.transform([news_text])

            # Predict with all models
            model_preds = {}
            model_confs = {}

            for name, model in models.items():
                pred, conf = get_prediction(model, X)
                model_preds[name] = pred
                model_confs[name] = conf

            # Majority vote
            final_pred, ones, zeros = majority_vote(model_preds)

            # Output
            badge = "<span class='badge-real'>REAL ✅</span>" if final_pred == 1 else "<span class='badge-fake'>FAKE ❌</span>"

            st.markdown(f"""
            <div class="card">
                <h3 style="margin:0;">Final Result: {badge}</h3>
                <p class="muted" style="margin:10px 0 0 0;">
                    Majority Vote → REAL votes: <b>{ones}</b> | FAKE votes: <b>{zeros}</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.markdown("### 🧠 Model-wise Predictions")

            for name in model_preds:
                pred_val = model_preds[name]
                result = "REAL ✅" if pred_val == 1 else "FAKE ❌"
                conf = model_confs[name]

                if conf is not None:
                    st.write(f"**{name}:** {result}  |  Confidence: **{conf:.2f}**")
                else:
                    st.write(f"**{name}:** {result}")

            st.write("")
            st.caption(f"🕒 Checked at: {datetime.now().strftime('%d %b %Y, %I:%M %p')}")

    else:
        st.info("Paste the news content and click **Detect News**.")


# ------------------ Footer ------------------
st.write("")
st.caption("⚡ Tip: Use full paragraph news content for best accuracy.")
