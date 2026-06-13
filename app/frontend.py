"""
KrishiMitra AI -- Streamlit Frontend
Mobile-friendly crop disease detection interface for Indian farmers.
"""

import streamlit as st
import requests
import io
from PIL import Image

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="KrishiMitra AI",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

API_URL = "http://localhost:8000"

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a6b1a, #2d9e2d);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
    }
    .disease-card {
        background: #fff3cd;
        border-left: 5px solid #ff9800;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .healthy-card {
        background: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .warning-card {
        background: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .rec-item {
        background: #f8f9fa;
        padding: 8px 12px;
        border-radius: 6px;
        margin: 5px 0;
        border-left: 3px solid #2d9e2d;
    }
    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 12px;
        margin-top: 30px;
        padding: 10px;
        border-top: 1px solid #eee;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1a6b1a, #2d9e2d);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 16px;
        width: 100%;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌱 KrishiMitra AI</h1>
    <p>Your AI farming expert — anytime, anywhere</p>
    <p><small>आपका AI कृषि विशेषज्ञ — कभी भी, कहीं भी</small></p>
</div>
""", unsafe_allow_html=True)


# ── Language Toggle ────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    lang_label = "🌐 Language / भाषा"
with col2:
    language = st.selectbox(
        lang_label,
        options=["en", "hi"],
        format_func=lambda x: "🇬🇧 English" if x == "en" else "🇮🇳 हिंदी",
        label_visibility="collapsed"
    )

# ── Language strings ───────────────────────────────────────
TEXT = {
    "en": {
        "upload_header":    "📸 Upload Crop Image",
        "upload_help":      "Take a clear photo of the affected leaf",
        "upload_label":     "Choose image (JPEG/PNG)",
        "query_label":      "💬 Your Question (optional)",
        "query_placeholder":"e.g. My tomato leaves have brown spots",
        "crop_label":       "🌾 Crop Name (optional)",
        "crop_placeholder": "e.g. tomato, potato, wheat",
        "analyze_btn":      "🔍 Analyze My Crop",
        "analyzing":        "🔄 Analyzing your crop...",
        "result_header":    "📊 Analysis Result",
        "disease_label":    "Detected Condition",
        "confidence_label": "Confidence",
        "severity_label":   "Severity",
        "advisory_header":  "💡 Advisory",
        "recs_header":      "✅ What To Do",
        "escalate_warn":    "⚠️ Low confidence — please consult your local KVK expert.",
        "disclaimer_label": "ℹ️ Disclaimer",
        "feedback_header":  "Was this helpful?",
        "thumbs_up":        "👍 Yes, helpful!",
        "thumbs_down":      "👎 Not helpful",
        "feedback_thanks":  "Thank you for your feedback!",
        "api_error":        "Could not connect to server. Please try again.",
        "healthy_msg":      "🎉 Great news! Your crop looks healthy!",
    },
    "hi": {
        "upload_header":    "📸 फसल की तस्वीर अपलोड करें",
        "upload_help":      "प्रभावित पत्ती की स्पष्ट तस्वीर लें",
        "upload_label":     "तस्वीर चुनें (JPEG/PNG)",
        "query_label":      "💬 आपका प्रश्न (वैकल्पिक)",
        "query_placeholder":"जैसे: मेरे टमाटर की पत्तियों पर भूरे धब्बे हैं",
        "crop_label":       "🌾 फसल का नाम (वैकल्पिक)",
        "crop_placeholder": "जैसे: टमाटर, आलू, गेहूं",
        "analyze_btn":      "🔍 फसल की जांच करें",
        "analyzing":        "🔄 फसल का विश्लेषण हो रहा है...",
        "result_header":    "📊 विश्लेषण परिणाम",
        "disease_label":    "पहचाना गया रोग",
        "confidence_label": "विश्वास स्तर",
        "severity_label":   "गंभीरता",
        "advisory_header":  "💡 सलाह",
        "recs_header":      "✅ क्या करें",
        "escalate_warn":    "⚠️ कम विश्वास — कृपया अपने स्थानीय KVK विशेषज्ञ से सलाह लें।",
        "disclaimer_label": "ℹ️ अस्वीकरण",
        "feedback_header":  "क्या यह सहायक था?",
        "thumbs_up":        "👍 हाँ, सहायक था!",
        "thumbs_down":      "👎 सहायक नहीं था",
        "feedback_thanks":  "आपकी प्रतिक्रिया के लिए धन्यवाद!",
        "api_error":        "सर्वर से कनेक्ट नहीं हो सका। कृपया पुनः प्रयास करें।",
        "healthy_msg":      "🎉 बधाई हो! आपकी फसल स्वस्थ दिखती है!",
    }
}
T = TEXT[language]


# ── Image Upload ───────────────────────────────────────────
st.markdown(f"### {T['upload_header']}")
st.caption(T["upload_help"])

uploaded_file = st.file_uploader(
    T["upload_label"],
    type=["jpg", "jpeg", "png", "webp"],
    help=T["upload_help"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# ── Query Input ────────────────────────────────────────────
query = st.text_input(
    T["query_label"],
    placeholder=T["query_placeholder"]
)

crop_name = st.text_input(
    T["crop_label"],
    placeholder=T["crop_placeholder"]
)

# ── Analyze Button ─────────────────────────────────────────
if st.button(T["analyze_btn"], disabled=uploaded_file is None):
    if uploaded_file is None:
        st.warning("Please upload an image first.")
    else:
        with st.spinner(T["analyzing"]):
            try:
                uploaded_file.seek(0)
                files = {"image": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                data  = {
                    "language":  language,
                    "query":     query or "",
                    "crop_name": crop_name or "",
                }
                response = requests.post(
                    f"{API_URL}/advisory",
                    files=files,
                    data=data,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    advisory_id = result["advisory_id"]

                    st.markdown(f"### {T['result_header']}")

                    # ── Disease Info Card ──────────────────
                    disease      = result["disease"]
                    is_healthy   = "healthy" in disease["disease_name"].lower()
                    card_class   = "healthy-card" if is_healthy else "disease-card"

                    if is_healthy:
                        st.markdown(f'<div class="{card_class}"><h3>{T["healthy_msg"]}</h3></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="{card_class}"><h3>🦠 {disease["disease_name"]}</h3></div>', unsafe_allow_html=True)

                    # ── Metrics ────────────────────────────
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric(T["confidence_label"], f"{disease['confidence']:.0%}")
                    with c2:
                        st.metric(T["severity_label"], disease.get("severity","—").title())
                    with c3:
                        st.metric("Crop", disease.get("affected_crop","—").title())

                    # ── Escalation Warning ─────────────────
                    if result["escalate_to_expert"]:
                        st.markdown(f'<div class="warning-card">{T["escalate_warn"]}</div>', unsafe_allow_html=True)

                    # ── Advisory Text ──────────────────────
                    st.markdown(f"### {T['advisory_header']}")
                    st.info(result["advisory_text"])

                    # ── Recommendations ────────────────────
                    st.markdown(f"### {T['recs_header']}")
                    for i, rec in enumerate(result["recommendations"], 1):
                        st.markdown(f'<div class="rec-item">**{i}.** {rec}</div>', unsafe_allow_html=True)

                    # ── Disclaimer ─────────────────────────
                    st.caption(f"{T['disclaimer_label']}: {result['disclaimer']}")

                    # ── Feedback ───────────────────────────
                    st.markdown(f"---\n### {T['feedback_header']}")
                    fb_col1, fb_col2 = st.columns(2)

                    with fb_col1:
                        if st.button(T["thumbs_up"], key="thumbs_up"):
                            requests.post(f"{API_URL}/feedback",
                                json={"advisory_id": advisory_id, "rating": 1})
                            st.success(T["feedback_thanks"])

                    with fb_col2:
                        if st.button(T["thumbs_down"], key="thumbs_down"):
                            requests.post(f"{API_URL}/feedback",
                                json={"advisory_id": advisory_id, "rating": 0})
                            st.success(T["feedback_thanks"])

                else:
                    st.error(f"Error {response.status_code}: {response.json().get('detail','Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error(T["api_error"])
            except Exception as e:
                st.error(f"Unexpected error: {e}")


# ── Footer ─────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🌱 KrishiMitra AI | SDG 2 — Zero Hunger | 1M1B AI for Sustainability<br>
    Powered by EfficientNet-B0 + RAG + IBM Granite | ICAR Knowledge Base
</div>
""", unsafe_allow_html=True)
