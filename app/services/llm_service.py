"""
KrishiMitra AI -- IBM Granite LLM Advisory Service
Generates farmer-friendly crop disease advisory using IBM Granite.
Falls back to rule-based advisory if IBM API is unavailable.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from typing import Optional, Dict, List
from app.core.config import settings
from app.core.logger import logger, audit_logger


# ── Prompt Templates ──────────────────────────────────────

ADVISORY_PROMPT_EN = """You are KrishiMitra AI, an expert agricultural advisor helping Indian farmers.
A farmer has uploaded a crop image and needs your help.

DETECTED DISEASE: {disease_name}
CONFIDENCE: {confidence:.0%}
CROP: {crop_name}
FARMER QUERY: {query}

RELEVANT KNOWLEDGE FROM ICAR:
{context}

Please provide a clear, simple advisory for the farmer. Use easy language (avoid jargon).
Structure your response as:
1. What is wrong with the crop (1-2 sentences)
2. How serious is it (mild/moderate/severe)
3. What to do immediately (2-3 action steps)
4. Preventive measures for future (1-2 points)

Keep it concise and practical. The farmer may have low literacy."""

ADVISORY_PROMPT_HI = """आप KrishiMitra AI हैं, एक विशेषज्ञ कृषि सलाहकार जो भारतीय किसानों की मदद करते हैं।
एक किसान ने फसल की तस्वीर अपलोड की है और उन्हें आपकी मदद चाहिए।

पहचाना गया रोग: {disease_name}
विश्वास स्तर: {confidence:.0%}
फसल: {crop_name}
किसान का प्रश्न: {query}

ICAR से प्रासंगिक जानकारी:
{context}

कृपया किसान के लिए सरल भाषा में सलाह दें।
अपना उत्तर इस प्रकार दें:
1. फसल में क्या समस्या है (1-2 वाक्य)
2. यह कितना गंभीर है (हल्का/मध्यम/गंभीर)
3. तुरंत क्या करें (2-3 कदम)
4. भविष्य के लिए बचाव (1-2 बिंदु)

सरल और व्यावहारिक रखें।"""


class LLMService:
    """
    IBM Granite LLM Service for advisory generation.
    Automatically falls back to rule-based if IBM unavailable.
    """

    def __init__(self):
        self.ibm_client = None
        self.is_ibm_ready = False
        self._try_init_ibm()

    def _try_init_ibm(self):
        """Try to initialize IBM Watson ML client."""
        try:
            if settings.IBM_API_KEY == "dummy_key":
                logger.warning("IBM API key not set -- using fallback advisory")
                return

            from ibm_watson_machine_learning.foundation_models import Model
            from ibm_watson_machine_learning.metanames import (
                GenTextParamsMetaNames as GenParams
            )

            credentials = {
                "url":    settings.IBM_URL,
                "apikey": settings.IBM_API_KEY,
            }
            params = {
                GenParams.MAX_NEW_TOKENS: 400,
                GenParams.TEMPERATURE:    0.3,
                GenParams.TOP_P:          0.9,
            }
            self.ibm_client = Model(
                model_id="ibm/granite-13b-chat-v2",
                params=params,
                credentials=credentials,
                project_id=settings.IBM_PROJECT_ID,
            )
            self.is_ibm_ready = True
            logger.info("IBM Granite LLM initialized successfully")

        except Exception as e:
            logger.warning(f"IBM Granite not available: {e} -- using fallback")
            self.is_ibm_ready = False

    def generate_advisory(
        self,
        disease_name:  str,
        confidence:    float,
        crop_name:     str,
        query:         str,
        context:       str,
        language:      str = "en",
        advisory_id:   str = "",
    ) -> Dict:
        """
        Generates farmer advisory using IBM Granite or fallback.

        Returns dict with advisory_text and recommendations list.
        """
        # Audit log every request (Responsible AI)
        audit_logger.info(
            f"advisory_id={advisory_id} | "
            f"disease={disease_name} | "
            f"confidence={confidence:.2f} | "
            f"crop={crop_name} | "
            f"lang={language}"
        )

        if self.is_ibm_ready:
            return self._generate_ibm(
                disease_name, confidence, crop_name, query, context, language
            )
        else:
            return self._generate_fallback(
                disease_name, confidence, crop_name, language
            )

    def _generate_ibm(
        self,
        disease_name: str,
        confidence:   float,
        crop_name:    str,
        query:        str,
        context:      str,
        language:     str,
    ) -> Dict:
        """Generate advisory using IBM Granite LLM."""
        try:
            template = ADVISORY_PROMPT_HI if language == "hi" else ADVISORY_PROMPT_EN
            prompt = template.format(
                disease_name=disease_name,
                confidence=confidence,
                crop_name=crop_name,
                query=query or "What is wrong with my crop?",
                context=context,
            )

            response = self.ibm_client.generate_text(prompt=prompt)
            logger.info("IBM Granite advisory generated successfully")

            return {
                "advisory_text":   response.strip(),
                "recommendations": self._extract_recs_from_context(context),
                "source":          "IBM Granite-13B",
            }

        except Exception as e:
            logger.error(f"IBM generation failed: {e} -- falling back")
            return self._generate_fallback(
                disease_name, confidence, crop_name, language
            )

    def _generate_fallback(
        self,
        disease_name: str,
        confidence:   float,
        crop_name:    str,
        language:     str,
    ) -> Dict:
        """
        Rule-based fallback advisory when IBM is unavailable.
        Ensures farmers always get a response.
        """
        is_healthy = "healthy" in disease_name.lower()
        severity   = self._estimate_severity(confidence)

        if language == "hi":
            return self._fallback_hindi(
                disease_name, crop_name, severity, is_healthy, confidence
            )
        return self._fallback_english(
            disease_name, crop_name, severity, is_healthy, confidence
        )

    def _fallback_english(
        self,
        disease_name: str,
        crop_name:    str,
        severity:     str,
        is_healthy:   bool,
        confidence:   float,
    ) -> Dict:
        if is_healthy:
            return {
                "advisory_text": (
                    f"Good news! Your {crop_name} crop appears healthy "
                    f"(confidence: {confidence:.0%}). "
                    f"Continue your current farming practices. "
                    f"Monitor weekly for early signs of disease."
                ),
                "recommendations": [
                    "Continue regular monitoring of your crop",
                    "Maintain proper spacing for air circulation",
                    "Use drip irrigation to keep leaves dry",
                    "Apply balanced fertilizer as per schedule",
                ],
                "source": "KrishiMitra Rule-Based Advisory",
            }

        clean_name = disease_name.replace("___", " - ").replace("_", " ")
        return {
            "advisory_text": (
                f"Your {crop_name} crop shows signs of {clean_name} "
                f"(confidence: {confidence:.0%}, severity: {severity}). "
                f"This is a fungal/bacterial disease that needs immediate attention. "
                f"Please follow the recommended steps below to protect your crop."
            ),
            "recommendations": [
                f"Remove and destroy all visibly infected {crop_name} leaves immediately",
                "Apply appropriate fungicide/bactericide as per ICAR guidelines",
                "Avoid overhead irrigation — switch to drip if possible",
                "Increase plant spacing to improve air circulation",
                f"Contact your local KVK if more than 30% of {crop_name} is affected",
            ],
            "source": "KrishiMitra Rule-Based Advisory",
        }

    def _fallback_hindi(
        self,
        disease_name: str,
        crop_name:    str,
        severity:     str,
        is_healthy:   bool,
        confidence:   float,
    ) -> Dict:
        if is_healthy:
            return {
                "advisory_text": (
                    f"अच्छी खबर! आपकी {crop_name} फसल स्वस्थ दिखती है "
                    f"(विश्वास: {confidence:.0%})। "
                    f"अपनी वर्तमान खेती की प्रथाएं जारी रखें।"
                ),
                "recommendations": [
                    "फसल की साप्ताहिक निगरानी जारी रखें",
                    "उचित दूरी बनाए रखें",
                    "ड्रिप सिंचाई का उपयोग करें",
                    "संतुलित खाद समय पर दें",
                ],
                "source": "KrishiMitra नियम-आधारित सलाह",
            }

        clean_name = disease_name.replace("___", " - ").replace("_", " ")
        return {
            "advisory_text": (
                f"आपकी {crop_name} फसल में {clean_name} के लक्षण हैं "
                f"(विश्वास: {confidence:.0%}, गंभीरता: {severity})। "
                f"कृपया तुरंत नीचे दिए गए कदम उठाएं।"
            ),
            "recommendations": [
                "रोगग्रस्त पत्तियां तुरंत तोड़कर जला दें",
                "ICAR दिशानिर्देशों के अनुसार फफूंदनाशक का छिड़काव करें",
                "ऊपर से सिंचाई बंद करें — ड्रिप सिंचाई अपनाएं",
                "पौधों के बीच दूरी बढ़ाएं",
                "30% से अधिक फसल प्रभावित हो तो KVK से संपर्क करें",
            ],
            "source": "KrishiMitra नियम-आधारित सलाह",
        }

    def _estimate_severity(self, confidence: float) -> str:
        if confidence >= 0.85:
            return "severe"
        elif confidence >= 0.65:
            return "moderate"
        return "mild"

    def _extract_recs_from_context(self, context: str) -> List[str]:
        """Extract recommendation sentences from RAG context."""
        recs = []
        for line in context.split("\n"):
            line = line.strip()
            if line and len(line) > 20:
                recs.append(line)
        return recs[:5] if recs else ["Follow ICAR guidelines for treatment."]


# Singleton instance
llm_service = LLMService()
