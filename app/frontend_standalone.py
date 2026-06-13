"""
KrishiMitra AI -- Streamlit Frontend (Standalone)
Works without separate FastAPI server for cloud deployment.
"""

import streamlit as st
import io
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from PIL import Image

st.set_page_config(
    page_title="KrishiMitra AI",
    page_icon="🌱",
    layout="centered",
)

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
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🌱 KrishiMitra AI</h1>
    <p>Your AI farming expert — anytime, anywhere</p>
    <p><small>आपका AI कृषि विशेषज्ञ — कभी भी, कहीं भी</small></p>
</div>
""", unsafe_allow_html=True)

language = st.selectbox(
    "🌐 Language",
    options=["en", "hi"],
    format_func=lambda x: "🇬🇧 English" if x == "en" else "🇮🇳 हिंदी"
)

st.markdown("### 📸 Upload Crop Image")
uploaded_file = st.file_uploader(
    "Choose image (JPEG/PNG)",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

query     = st.text_input("💬 Your Question (optional)", placeholder="e.g. My tomato leaves have brown spots")
crop_name = st.text_input("🌾 Crop Name (optional)", placeholder="e.g. tomato, potato")

if st.button("🔍 Analyze My Crop", disabled=uploaded_file is None):
    with st.spinner("🔄 Analyzing your crop..."):
        try:
            from app.utils.image_preprocessor import preprocess_image
            from app.api.routes.advisory import get_classifier
            from app.services.rag_service import rag_service
            from app.services.llm_service import llm_service

            uploaded_file.seek(0)
            image_bytes = uploaded_file.read()
            tensor = preprocess_image(image_bytes)

            if tensor is None:
                st.error("Invalid image. Please upload a clear leaf photo.")
            else:
                clf = get_classifier()
                if not rag_service.is_ready:
                    rag_service.initialize()

                prediction = clf.predict(tensor)
                disease_name  = prediction["disease_name"]
                confidence    = prediction["confidence"]
                escalate      = prediction["escalate_to_expert"]
                detected_crop = crop_name or clf.get_crop_from_class(prediction["disease_class"])
                is_healthy    = prediction["is_healthy"]

                context = rag_service.get_context_text(
                    query=query or disease_name,
                    disease_name=disease_name,
                    crop_name=detected_crop,
                    language=language,
                )

                llm_result = llm_service.generate_advisory(
                    disease_name=disease_name,
                    confidence=confidence,
                    crop_name=detected_crop,
                    query=query or "What is wrong with my crop?",
                    context=context,
                    language=language,
                    advisory_id="streamlit-cloud",
                )

                st.markdown("### 📊 Analysis Result")

                if is_healthy:
                    st.success(f"🎉 Your crop looks **healthy**!")
                else:
                    st.warning(f"🦠 Detected: **{disease_name}**")

                c1, c2, c3 = st.columns(3)
                c1.metric("Confidence", f"{confidence:.0%}")
                c2.metric("Severity", "Severe" if confidence >= 0.85 else "Moderate" if confidence >= 0.65 else "Mild")
                c3.metric("Crop", detected_crop.title())

                if escalate:
                    st.error("⚠️ Low confidence — consult your local KVK expert.")

                st.markdown("### 💡 Advisory")
                st.info(llm_result["advisory_text"])

                st.markdown("### ✅ What To Do")
                for i, rec in enumerate(llm_result["recommendations"], 1):
                    st.markdown(f"**{i}.** {rec}")

                st.caption("ℹ️ This advisory is AI-generated. For severe cases, consult your local KVK.")

                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Helpful!"):
                        st.success("Thank you!")
                with col2:
                    if st.button("👎 Not helpful"):
                        st.success("Thank you for your feedback!")

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure the knowledge base is ingested: `python scripts/ingest_knowledge.py`")

st.markdown("""
<div style='text-align:center;color:#888;font-size:12px;margin-top:30px'>
🌱 KrishiMitra AI | SDG 2 Zero Hunger | 1M1B AI for Sustainability
</div>
""", unsafe_allow_html=True)
