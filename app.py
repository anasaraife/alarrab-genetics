# ==============================================================================
#  العرّاب للجينات - الإصدار 13.0 المُطوَّر (مع تحسينات شاملة)
# ==============================================================================

import streamlit as st
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import gdown
import PyPDF2
import os
import tempfile
import requests
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import logging
from datetime import datetime
import hashlib
import re

# -------------------------------------------------
#  1. إعدادات الصفحة والمصادر المحدثة
# -------------------------------------------------
st.set_page_config(
    page_title="العرّاب للجينات - الإصدار 13.0 المُطوَّر",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# إعداد نظام السجلات
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOOK_LINKS = [
    "https://drive.google.com/file/d/1CRwW78pd2RsKVd37elefz71RqwaCaute/view?usp=sharing",
    "https://drive.google.com/file/d/1894OOW1nEc3SkanLKKEzaXu_XhXYv8rF/view?usp=sharing",
]

# قاموس المرادفات والكلمات المفتاحية
GENETICS_KEYWORDS = {
    "ألوان": ["لون", "أحمر", "أزرق", "أبيض", "أسود", "بني", "رمادي", "صبغة"],
    "وراثة": ["جين", "كروموسوم", "DNA", "صفة", "مندل", "هجين", "نقي"],
    "تربية": ["تزاوج", "انتقاء", "سلالة", "نسل", "جيل", "تهجين"],
    "طفرات": ["طفرة", "تحور", "شاذ", "نادر", "استثنائي"],
    "سلوك": ["طيران", "عودة", "توجه", "غذاء", "تغريد"]
}

# -------------------------------------------------
#  2. مدير نماذج الذكاء الاصطناعي المُطوَّر
# -------------------------------------------------
class AdvancedAIModelManager:
    def __init__(self):
        self.models = {
            "gemini": {
                "name": "Google Gemini Flash 1.5", 
                "available": False, 
                "priority": 1,
                "status": "فحص...",
                "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
                "last_check": None,
                "error_count": 0
            },
            "deepseek": {
                "name": "DeepSeek Chat", 
                "available": False, 
                "priority": 2,
                "status": "فحص...",
                "endpoint": "https://api.deepseek.com/v1/chat/completions",
                "last_check": None,
                "error_count": 0
            },
            "huggingface": {
                "name": "Hugging Face Inference", 
                "available": False, 
                "priority": 3,
                "status": "فحص...",
                "endpoint": "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium",
                "last_check": None,
                "error_count": 0
            },
            "fallback": {
                "name": "النمط الاحتياطي الذكي", 
                "available": True, 
                "priority": 4,
                "status": "دائماً متاح",
                "last_check": datetime.now(),
                "error_count": 0
            }
        }
        self._check_all_models()

    def _check_all_models(self):
        """فحص جميع النماذج مع تحديث الحالة"""
        with st.spinner("فحص النماذج المتاحة..."):
            for model_key in ["gemini", "deepseek", "huggingface"]:
                self._check_single_model(model_key)

    def _check_single_model(self, model_key: str):
        """فحص نموذج واحد مع معالجة شاملة للأخطاء"""
        try:
            if model_key == "gemini":
                key = st.secrets.get("GEMINI_API_KEY", "")
                if key and len(key) > 20:
                    # اختبار بسيط للاتصال
                    self.models[model_key]["available"] = True
                    self.models[model_key]["status"] = "✅ جاهز ومتصل"
                else:
                    self.models[model_key]["status"] = "❌ مفتاح API مفقود أو غير صحيح"
                    
            elif model_key == "deepseek":
                key = st.secrets.get("DEEPSEEK_API_KEY", "")
                if key and len(key) > 20:
                    self.models[model_key]["available"] = True
                    self.models[model_key]["status"] = "✅ جاهز ومتصل"
                else:
                    self.models[model_key]["status"] = "💡 مفتاح API غير موجود (اختياري)"
                    
            elif model_key == "huggingface":
                key = st.secrets.get("HUGGINGFACE_API_KEY", "")
                if key and len(key) > 20:
                    self.models[model_key]["available"] = True
                    self.models[model_key]["status"] = "✅ جاهز ومتصل"
                else:
                    self.models[model_key]["status"] = "💡 مفتاح API غير موجود (اختياري)"
                    
            self.models[model_key]["last_check"] = datetime.now()
            
        except Exception as e:
            self.models[model_key]["status"] = f"❌ خطأ في الفحص: {str(e)[:50]}..."
            self.models[model_key]["error_count"] += 1
            logger.error(f"خطأ في فحص {model_key}: {e}")

    def get_available_models(self) -> List[str]:
        """الحصول على قائمة النماذج المتاحة مرتبة حسب الأولوية"""
        available = [model for model, config in self.models.items() if config["available"]]
        return sorted(available, key=lambda x: self.models[x]["priority"])

    def get_model_stats(self) -> Dict:
        """إحصائيات مفصلة عن النماذج"""
        return {
            "total": len(self.models),
            "available": len([m for m in self.models.values() if m["available"]]),
            "errors": sum([m["error_count"] for m in self.models.values()]),
            "last_update": max([m.get("last_check", datetime.now()) for m in self.models.values()])
        }

# -------------------------------------------------
#  3. نظام ذاكرة التخزين المؤقت المحسن
# -------------------------------------------------
class SmartCache:
    def __init__(self):
        self.cache_dir = os.path.join(tempfile.gettempdir(), "genetics_cache_v13")
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_key(self, content: str) -> str:
        """إنشاء مفتاح فريد للمحتوى"""
        return hashlib.md5(content.encode()).hexdigest()

    def cache_response(self, query: str, response: str, source: str):
        """حفظ الاستجابة في الذاكرة المؤقتة"""
        try:
            cache_key = self.get_cache_key(query)
            cache_data = {
                "query": query,
                "response": response,
                "source": source,
                "timestamp": datetime.now().isoformat(),
                "hash": cache_key
            }
            
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"خطأ في حفظ الذاكرة المؤقتة: {e}")

    def get_cached_response(self, query: str) -> Optional[Dict]:
        """البحث عن استجابة محفوظة"""
        try:
            cache_key = self.get_cache_key(query)
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                # فحص عمر الذاكرة المؤقتة (24 ساعة)
                cache_time = datetime.fromisoformat(cache_data["timestamp"])
                if (datetime.now() - cache_time).seconds < 86400:
                    return cache_data
                    
        except Exception as e:
            logger.error(f"خطأ في قراءة الذاكرة المؤقتة: {e}")
        
        return None

# -------------------------------------------------
#  4. محرك البحث المطور
# -------------------------------------------------
@st.cache_resource
def load_advanced_embedding_model():
    """تحميل نموذج التضمين مع معالجة متقدمة"""
    try:
        with st.spinner("تحميل نموذج التضمين المتقدم..."):
            model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            st.success("✅ تم تحميل نموذج التضمين بنجاح")
            return model
    except Exception as e:
        st.error(f"❌ خطأ في تحميل نموذج التضمين: {e}")
        return None

@st.cache_data(ttl=86400, show_spinner=False)
def build_advanced_knowledge_base(_model):
    """بناء قاعدة معرفة متطورة مع فهرسة محسنة"""
    if _model is None:
        return None
        
    db_path = os.path.join(tempfile.gettempdir(), "advanced_genetics_kb_v13.db")
    
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.cursor()
        
        # إنشاء جداول محسنة
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                source TEXT,
                content TEXT UNIQUE,
                content_hash TEXT,
                keywords TEXT,
                page_number INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        
        if doc_count == 0:
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_pages = 0
                for i, link in enumerate(BOOK_LINKS):
                    try:
                        status_text.text(f"📖 تحميل وتحليل الكتاب {i+1} من {len(BOOK_LINKS)}...")
                        progress_bar.progress(i / len(BOOK_LINKS))
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                            file_id = link.split('/d/')[1].split('/')[0]
                            gdown.download(id=file_id, output=tmp.name, quiet=True)
                            
                            with open(tmp.name, 'rb') as f:
                                reader = PyPDF2.PdfReader(f)
                                
                                for page_num, page in enumerate(reader.pages):
                                    text = page.extract_text() or ""
                                    cleaned_text = clean_and_enhance_text(text)
                                    
                                    if len(cleaned_text.strip()) > 100:
                                        # استخراج الكلمات المفتاحية
                                        keywords = extract_keywords(cleaned_text)
                                        content_hash = hashlib.md5(cleaned_text.encode()).hexdigest()
                                        
                                        cursor.execute("""
                                            INSERT OR IGNORE INTO documents 
                                            (source, content, content_hash, keywords, page_number) 
                                            VALUES (?, ?, ?, ?, ?)
                                        """, (
                                            f"الكتاب {i+1}، الصفحة {page_num+1}",
                                            cleaned_text.strip(),
                                            content_hash,
                                            ", ".join(keywords),
                                            page_num + 1
                                        ))
                                        total_pages += 1
                            
                            os.remove(tmp.name)
                            
                    except Exception as e:
                        st.warning(f"⚠️ تعذر معالجة الكتاب {i+1}: {e}")
                        continue
                
                conn.commit()
                progress_bar.progress(1.0)
                status_text.text(f"✅ تم الانتهاء! تمت معالجة {total_pages} صفحة")
                time.sleep(2)
                progress_container.empty()

        # استرجاع جميع الوثائق
        cursor.execute("SELECT id, source, content, keywords FROM documents ORDER BY id")
        all_docs = [
            {
                "id": row[0], 
                "source": row[1], 
                "content": row[2], 
                "keywords": row[3] or ""
            } 
            for row in cursor.fetchall()
        ]
        conn.close()

        if not all_docs:
            st.warning("⚠️ قاعدة المعرفة فارغة")
            return None
        
        st.success(f"✅ قاعدة المعرفة: {len(all_docs)} وثيقة جاهزة")
        
        # إنشاء التضمينات
        with st.spinner("إنشاء فهرس البحث الدلالي..."):
            contents = [doc['content'] for doc in all_docs]
            embeddings = _model.encode(contents, show_progress_bar=False, batch_size=32)
        
        return {"documents": all_docs, "embeddings": embeddings}
        
    except Exception as e:
        st.error(f"❌ خطأ في بناء قاعدة المعرفة: {e}")
        return None

def clean_and_enhance_text(text: str) -> str:
    """تنظيف وتحسين النص المستخرج"""
    # إزالة الأسطر الفارغة والمسافات الزائدة
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    # إزالة الرموز غير المرغوبة
    text = re.sub(r'[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', ' ', text)
    
    return text.strip()

def extract_keywords(text: str) -> List[str]:
    """استخراج الكلمات المفتاحية من النص"""
    keywords = set()
    text_lower = text.lower()
    
    for category, words in GENETICS_KEYWORDS.items():
        for word in words:
            if word in text_lower:
                keywords.add(category)
                keywords.add(word)
    
    return list(keywords)

def advanced_semantic_search(query: str, model, knowledge_base, limit=7):
    """بحث دلالي متطور مع تحسينات"""
    if not knowledge_base or not model:
        return []
    
    try:
        # تحسين الاستعلام
        enhanced_query = enhance_query(query)
        
        # البحث الدلالي
        query_embedding = model.encode([enhanced_query])
        similarities = cosine_similarity(query_embedding, knowledge_base['embeddings'])[0]
        
        # البحث بالكلمات المفتاحية
        keyword_matches = []
        for i, doc in enumerate(knowledge_base['documents']):
            keyword_score = calculate_keyword_match(query, doc.get('keywords', ''))
            if keyword_score > 0:
                keyword_matches.append((i, keyword_score))
        
        # دمج النتائج
        combined_scores = {}
        
        # النتائج الدلالية
        semantic_indices = np.argsort(similarities)[-limit*2:][::-1]
        for idx in semantic_indices:
            if similarities[idx] > 0.2:  # عتبة مرونة أكثر
                combined_scores[idx] = similarities[idx] * 0.7
        
        # إضافة نتائج الكلمات المفتاحية
        for idx, keyword_score in keyword_matches:
            if idx in combined_scores:
                combined_scores[idx] += keyword_score * 0.3
            else:
                combined_scores[idx] = keyword_score * 0.3
        
        # ترتيب النتائج النهائية
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, score in sorted_results[:limit]]
        
        return [knowledge_base['documents'][i] for i in top_indices]
        
    except Exception as e:
        st.error(f"خطأ في البحث المتطور: {e}")
        return []

def enhance_query(query: str) -> str:
    """تحسين الاستعلام بإضافة مرادفات"""
    enhanced = query
    query_lower = query.lower()
    
    for category, words in GENETICS_KEYWORDS.items():
        for word in words:
            if word in query_lower:
                # إضافة مرادفات من نفس الفئة
                other_words = [w for w in words if w != word]
                enhanced += f" {' '.join(other_words[:3])}"
                break
    
    return enhanced

def calculate_keyword_match(query: str, keywords: str) -> float:
    """حساب مطابقة الكلمات المفتاحية"""
    if not keywords:
        return 0.0
    
    query_words = set(query.lower().split())
    keyword_list = set(keywords.lower().split(', '))
    
    intersection = query_words.intersection(keyword_list)
    if not intersection:
        return 0.0
    
    return len(intersection) / max(len(query_words), len(keyword_list))

# -------------------------------------------------
#  5. نظام الردود الذكية المطور
# -------------------------------------------------
class IntelligentResponseSystem:
    def __init__(self, ai_manager: AdvancedAIModelManager):
        self.ai_manager = ai_manager
        self.cache = SmartCache()
        self.available_models = ai_manager.get_available_models()

    def get_gemini_response(self, query: str, context_docs: List[Dict]) -> Tuple[str, bool, str]:
        """استجابة Gemini محسنة مع معالجة شاملة"""
        try:
            # فحص الذاكرة المؤقتة أولاً
            cache_key = f"gemini_{query}_{len(context_docs)}"
            cached = self.cache.get_cached_response(cache_key)
            if cached:
                return cached["response"], True, "من الذاكرة المؤقتة"

            API_KEY = st.secrets.get("GEMINI_API_KEY", "")
            if not API_KEY:
                return "مفتاح Gemini API غير موجود", False, "خطأ في الإعداد"
                
            API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
            
            # إعداد السياق المحسن
            if context_docs:
                # تحسين ترتيب الوثائق
                sorted_docs = sorted(context_docs, key=lambda x: len(x.get('keywords', '')), reverse=True)
                context_parts = []
                
                for i, doc in enumerate(sorted_docs[:5]):  # أفضل 5 وثائق
                    context_part = f"""📖 المرجع {i+1}: {doc['source']}
🔑 الكلمات المفتاحية: {doc.get('keywords', 'غير متوفرة')}
📝 المحتوى: {doc['content'][:600]}...

"""
                    context_parts.append(context_part)
                
                context = "\n".join(context_parts)
                
                prompt = f"""أنت العرّاب، خبير وراثة الحمام الأول عربياً. مهمتك الإجابة بدقة علمية ووضوح تام.

📚 المراجع العلمية المتوفرة:
{context}

❓ سؤال المربي: {query}

📋 متطلبات الإجابة:
• استخدم المراجع المتوفرة أعلاه فقط
• اذكر المصادر عند الحاجة
• اجعل الإجابة علمية لكن مفهومة
• استخدم الرموز التعبيرية للوضوح
• قدم أمثلة عملية إن أمكن

🔬 الإجابة الخبيرة:"""
            else:
                prompt = f"""أنت العرّاب، خبير وراثة الحمام الأول عربياً.

❓ سؤال المربي: {query}

🔬 إجابتك الخبيرة (بالعربية، علمية ووافية):"""

            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 1200,
                    "topP": 0.9,
                    "topK": 40
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}
                ]
            }
            
            response = requests.post(
                API_URL, 
                json=payload, 
                headers={"Content-Type": "application/json"},
                timeout=45
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    answer = result['candidates'][0]['content']['parts'][0]['text']
                    
                    # حفظ في الذاكرة المؤقتة
                    self.cache.cache_response(cache_key, answer, "Gemini")
                    
                    return answer, True, "Gemini Flash 1.5"
                else:
                    return "استجابة غير مكتملة من Gemini", False, "خطأ في الاستجابة"
            else:
                error_detail = f"HTTP {response.status_code}"
                if response.status_code == 429:
                    error_detail += " - تم تجاوز حد الطلبات"
                elif response.status_code == 403:
                    error_detail += " - مشكلة في صلاحيات API"
                return f"خطأ Gemini: {error_detail}", False, "خطأ شبكة"
                
        except requests.exceptions.Timeout:
            return "انتهت مهلة الاتصال مع Gemini (45 ثانية)", False, "انتهاء مهلة"
        except requests.exceptions.RequestException as e:
            return f"خطأ شبكة: {str(e)[:100]}...", False, "خطأ اتصال"
        except Exception as e:
            return f"خطأ غير متوقع في Gemini: {str(e)[:100]}...", False, "خطأ عام"

    def get_intelligent_fallback(self, query: str, context_docs: List[Dict] = None) -> str:
        """نظام احتياطي ذكي محسن"""
        
        # تحليل السؤال
        query_analysis = self.analyze_query(query)
        
        # بناء إجابة ذكية بناءً على التحليل
        response_parts = ["🧬 **العرّاب - النمط الاحتياطي الذكي**\n"]
        
        if context_docs:
            response_parts.append(f"📚 تم العثور على {len(context_docs)} مرجع ذي صلة:")
            for i, doc in enumerate(context_docs[:3]):
                response_parts.append(f"• {doc['source']}")
            response_parts.append("")
        
        # إجابة مخصصة حسب نوع السؤال
        if query_analysis["category"] == "colors":
            response_parts.append(self._get_color_genetics_response(query_analysis["keywords"]))
        elif query_analysis["category"] == "breeding":
            response_parts.append(self._get_breeding_response(query_analysis["keywords"]))
        elif query_analysis["category"] == "genetics":
            response_parts.append(self._get_genetics_response(query_analysis["keywords"]))
        elif query_analysis["category"] == "behavior":
            response_parts.append(self._get_behavior_response(query_analysis["keywords"]))
        else:
            response_parts.append(self._get_general_response(query))
        
        response_parts.extend([
            "\n---",
            "💡 **ملاحظة**: هذه إجابة من النظام الاحتياطي الذكي.",
            "للحصول على إجابات أكثر تفصيلاً، يرجى التأكد من إعداد مفاتيح API."
        ])
        
        return "\n".join(response_parts)

    def analyze_query(self, query: str) -> Dict:
        """تحليل السؤال لتحديد الفئة والكلمات المفتاحية"""
        query_lower = query.lower()
        analysis = {
            "category": "general",
            "keywords": [],
            "complexity": "medium"
        }
        
        # تحديد الفئة
        color_words = ["لون", "أحمر", "أزرق", "أبيض", "أسود", "بني", "رمادي", "صبغة", "ألوان"]
        breeding_words = ["تربية", "تزاوج", "انتقاء", "سلالة", "نسل", "جيل", "تهجين"]
        genetics_words = ["وراثة", "جين", "كروموسوم", "DNA", "صفة", "مندل", "هجين", "نقي"]
        behavior_words = ["سلوك", "طيران", "عودة", "توجه", "غذاء", "تغريد"]
        
        if any(word in query_lower for word in color_words):
            analysis["category"] = "colors"
            analysis["keywords"] = [word for word in color_words if word in query_lower]
        elif any(word in query_lower for word in breeding_words):
            analysis["category"] = "breeding"
            analysis["keywords"] = [word for word in breeding_words if word in query_lower]
        elif any(word in query_lower for word in genetics_words):
            analysis["category"] = "genetics"
            analysis["keywords"] = [word for word in genetics_words if word in query_lower]
        elif any(word in query_lower for word in behavior_words):
            analysis["category"] = "behavior"
            analysis["keywords"] = [word for word in behavior_words if word in query_lower]
        
        # تحديد التعقيد
        if len(query.split()) > 15 or "كيف" in query_lower or "لماذا" in query_lower:
            analysis["complexity"] = "high"
        elif len(query.split()) < 5:
            analysis["complexity"] = "low"
        
        return analysis

    def _get_color_genetics_response(self, keywords: List[str]) -> str:
        """إجابة متخصصة في وراثة الألوان"""
        return """🎨 **وراثة الألوان في الحمام**

الألوان في الحمام تحكمها عدة جينات رئيسية:

🔹 **الجينات الأساسية:**
• جين B: يحدد اللون الأساسي (أزرق/بني/أحمر)
• جين C: يتحكم في شدة اللون
• جين D: يؤثر على تشبع اللون

🔹 **أنماط الوراثة:**
• الأزرق: الصفة السائدة الأكثر شيوعاً
• الأحمر: مرتبط بالكروموسوم الجنسي
• البني: صفة متنحية تحتاج جينين متماثلين

🔹 **التفاعلات الجينية:**
• تفاعل عدة جينات ينتج تدرجات لونية مختلفة
• الطفرات قد تنتج ألوان نادرة وجميلة"""

    def _get_breeding_response(self, keywords: List[str]) -> str:
        """إجابة متخصصة في التربية"""
        return """🐦 **أسس التربية الناجحة**

🔹 **الانتقاء الصحيح:**
• اختيار الأبوين بناءً على الصفات المرغوبة
• تجنب زواج الأقارب المفرط
• مراعاة التوازن بين الشكل والأداء

🔹 **التخطيط الوراثي:**
• فهم الصفات السائدة والمتنحية
• التنبؤ بصفات النسل
• الاحتفاظ بسجلات دقيقة

🔹 **العناية بالنسل:**
• توفير بيئة مناسبة للتكاثر
• التغذية المتوازنة للأبوين
• مراقبة صحة الفراخ الصغيرة"""

    def _get_genetics_response(self, keywords: List[str]) -> str:
        """إجابة متخصصة في علم الوراثة"""
        return """🧬 **أساسيات علم الوراثة**

🔹 **المفاهيم الأساسية:**
• الكروموسومات: تحمل المعلومات الوراثية
• الجينات: وحدات الوراثة الأساسية
• الأليلات: صور مختلفة للجين الواحد

🔹 **قوانين مندل:**
• قانون الانعزال: كل صفة تتحكم فيها عوامل منفصلة
• قانون التوزيع المستقل: الصفات المختلفة تورث بشكل مستقل
• السيادة والتنحي: بعض الصفات تغطي أخرى

🔹 **التطبيق العملي:**
• استخدام مربعات بونيت للتنبؤ
• فهم الوراثة المرتبطة بالجنس
• التعامل مع الصفات متعددة الجينات"""

    def _get_behavior_response(self, keywords: List[str]) -> str:
        """إجابة متخصصة في السلوك"""
        return """🕊️ **سلوك الحمام وعلم الوراثة**

🔹 **السلوكيات الموروثة:**
• قدرة العودة للمنزل (الهومينغ)
• أنماط الطيران المختلفة
• سلوك التودد والتزاوج

🔹 **العوامل الوراثية:**
• بعض السلوكيات تحكمها جينات محددة
• التفاعل بين الوراثة والبيئة
• إمكانية تحسين السلوك بالانتقاء

🔹 **التطبيق في التربية:**
• انتقاء الطيور ذات السلوك المرغوب
• تجنب السلوكيات العدوانية المفرطة
• تطوير خطوط وراثية متخصصة"""

    def _get_general_response(self, query: str) -> str:
        """إجابة عامة ذكية"""
        return f"""🔍 **حول استفسارك: "{query[:50]}..."**

🔹 **ما يمكنني مساعدتك فيه:**
• وراثة الألوان والأنماط في الحمام
• أسس التربية والتهجين الصحيح
• شرح المفاهيم الوراثية الأساسية
• السلوك الموروث في الحمام
• حل مشاكل التربية الشائعة

🔹 **نصائح للحصول على إجابة أفضل:**
• حدد نوع المشكلة أو السؤال بوضوح
• اذكر تفاصيل عن طيورك إن أمكن
• استخدم كلمات مفتاحية واضحة

💡 مثال: "ما وراثة اللون الأحمر في الحمام؟" """

    def get_comprehensive_answer(self, query: str, context_docs: List[Dict]) -> Tuple[str, str, str]:
        """الحصول على إجابة شاملة ومحسنة"""
        
        # فحص الذاكرة المؤقتة أولاً
        cached_response = self.cache.get_cached_response(query)
        if cached_response:
            return cached_response["response"], cached_response["source"], "مخزن مؤقتاً"
        
        # محاولة Gemini مع السياق
        if context_docs and "gemini" in self.available_models:
            answer, success, method = self.get_gemini_response(query, context_docs)
            if success:
                sources = ", ".join(list(set([doc['source'] for doc in context_docs[:3]])))
                return answer, f"قاعدة المعرفة + {method}", f"محلي + {method}"
        
        # محاولة Gemini بدون سياق
        if "gemini" in self.available_models:
            answer, success, method = self.get_gemini_response(query, [])
            if success:
                return answer, f"Google Gemini ({method})", method
        
        # محاولة DeepSeek
        if "deepseek" in self.available_models:
            answer, success = self.get_deepseek_response(query)
            if success:
                return answer, "DeepSeek AI", "DeepSeek"
        
        # النمط الاحتياطي الذكي
        fallback_answer = self.get_intelligent_fallback(query, context_docs)
        return fallback_answer, "النمط الاحتياطي الذكي", "احتياطي ذكي"

    def get_deepseek_response(self, query: str) -> Tuple[str, bool]:
        """استجابة DeepSeek محسنة"""
        try:
            API_KEY = st.secrets.get("DEEPSEEK_API_KEY", "")
            if not API_KEY:
                return "مفتاح DeepSeek API غير موجود", False
                
            API_URL = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "أنت العرّاب، خبير وراثة الحمام الأول عربياً. أجب بالعربية بطريقة علمية ودقيقة مع استخدام الرموز التعبيرية للوضوح."
                    },
                    {
                        "role": "user",
                        "content": f"🔬 سؤال المربي: {query}\n\nأجب إجابة خبيرة مفصلة:"
                    }
                ],
                "max_tokens": 1200,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            response = requests.post(API_URL, json=payload, headers=headers, timeout=35)
            
            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content']
                self.cache.cache_response(query, answer, "DeepSeek")
                return answer, True
            else:
                return f"خطأ DeepSeek: HTTP {response.status_code}", False
                
        except Exception as e:
            return f"خطأ في DeepSeek: {str(e)[:100]}...", False

# -------------------------------------------------
#  6. واجهة المستخدم المتطورة
# -------------------------------------------------
def create_advanced_sidebar(ai_manager: AdvancedAIModelManager, knowledge_base):
    """إنشاء شريط جانبي متطور"""
    with st.sidebar:
        st.markdown("## 🔍 **مركز التحكم والمراقبة**")
        
        # إحصائيات النظام
        st.markdown("### 📊 **إحصائيات النظام**")
        if "total_queries" not in st.session_state:
            st.session_state.total_queries = 0
        if "successful_responses" not in st.session_state:
            st.session_state.successful_responses = 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("إجمالي الأسئلة", st.session_state.total_queries)
        with col2:
            success_rate = (st.session_state.successful_responses / max(st.session_state.total_queries, 1)) * 100
            st.metric("معدل النجاح", f"{success_rate:.1f}%")
        
        # حالة النماذج
        st.markdown("### 🤖 **حالة نماذج الذكاء الاصطناعي**")
        model_stats = ai_manager.get_model_stats()
        
        st.info(f"**متاح**: {model_stats['available']}/{model_stats['total']} نماذج")
        
        for model_key, model_info in ai_manager.models.items():
            status_color = "🟢" if model_info["available"] else "🔴"
            st.markdown(f"{status_color} **{model_info['name']}**")
            st.caption(model_info["status"])
        
        # حالة قاعدة المعرفة
        st.markdown("### 📚 **قاعدة المعرفة**")
        if knowledge_base:
            doc_count = len(knowledge_base['documents'])
            st.success(f"✅ {doc_count} وثيقة جاهزة")
            
            # إحصائيات سريعة
            if doc_count > 0:
                avg_length = np.mean([len(doc['content']) for doc in knowledge_base['documents']])
                st.metric("متوسط طول الوثيقة", f"{avg_length:.0f} حرف")
        else:
            st.error("❌ قاعدة المعرفة غير متاحة")
        
        # أدوات التحكم
        st.markdown("### ⚙️ **أدوات التحكم**")
        
        if st.button("🔄 تحديث حالة النماذج"):
            ai_manager._check_all_models()
            st.rerun()
        
        if st.button("🗑️ مسح المحادثة"):
            st.session_state.messages = []
            st.rerun()
        
        # معلومات النسخة
        st.markdown("---")
        st.markdown("### ℹ️ **معلومات النسخة**")
        st.caption("🧬 العرّاب للجينات v13.0")
        st.caption("⚡ محرك ذكي متطور")
        st.caption("🔄 آخر تحديث: 2024")
        
        # نصائح سريعة
        with st.expander("💡 نصائح للحصول على أفضل إجابة"):
            st.markdown("""
            • **كن محدداً**: اذكر تفاصيل السؤال بوضوح
            • **استخدم كلمات مفتاحية**: مثل "وراثة"، "لون"، "تربية"
            • **اسأل سؤالاً واحداً**: لتحصل على إجابة مركزة
            • **اذكر نوع الحمام**: إن كان لديك سلالة معينة
            """)

def create_welcome_message() -> str:
    """إنشاء رسالة ترحيب ديناميكية"""
    return """🧬 **مرحباً بك في العرّاب للجينات v13.0 المُطوَّر!**

### 🆕 **المميزات الجديدة:**
- 🧠 **ذكاء اصطناعي متعدد المصادر** (Gemini، DeepSeek، Hugging Face)
- 🔍 **بحث دلالي متطور** مع فهرسة ذكية للكلمات المفتاحية
- 💾 **نظام ذاكرة مؤقتة ذكي** لإجابات أسرع
- 📊 **تشخيص شامل** مع مراقبة حالة النظام
- 🎯 **نمط احتياطي ذكي** يعمل حتى بدون اتصال API

### 🔬 **ما يمكنني مساعدتك فيه:**
• **وراثة الألوان**: كيف تنتقل الألوان في الحمام؟
• **التربية الانتقائية**: كيف تحسن سلالتك؟
• **حل المشاكل الوراثية**: لماذا ظهر هذا اللون؟
• **التخطيط للتزاوج**: ما أفضل اقتران؟
• **فهم الطفرات**: ما هذا الشكل الغريب؟

🚀 **جرب الآن!** اسأل أي سؤال عن وراثة الحمام وسأقدم لك إجابة خبيرة مفصلة!

---
💡 *نصيحة: ابدأ بسؤال محدد مثل "كيف أحصل على حمام أحمر اللون؟"*"""

def main():
    """الوظيفة الرئيسية المطورة"""
    # الهيدر والعنوان
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1>🧬 العرّاب للجينات</h1>
        <h3>الإصدار 13.0 المُطوَّر - خبير الوراثة الذكي</h3>
        <p style="color: #666;">نظام ذكي متعدد المصادر لخبرة وراثة الحمام</p>
    </div>
    """, unsafe_allow_html=True)

    # تحميل النماذج والأنظمة
    with st.spinner("🚀 تهيئة الأنظمة المتطورة..."):
        model = load_advanced_embedding_model()
        ai_manager = AdvancedAIModelManager()
        knowledge_base = build_advanced_knowledge_base(model) if model else None
        response_system = IntelligentResponseSystem(ai_manager)

    # الشريط الجانبي
    create_advanced_sidebar(ai_manager, knowledge_base)

    # إعداد المحادثة
    if "messages" not in st.session_state:
        welcome_msg = create_welcome_message()
        st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]

    # عرض المحادثة
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # معالجة الإدخال الجديد
    if prompt := st.chat_input("💬 اسأل العرّاب عن أي شيء متعلق بوراثة الحمام..."):
        # إضافة السؤال
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.total_queries += 1
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔍 العرّاب يبحث في قاعدة المعرفة ويستشير خبراء الذكاء الاصطناعي..."):
                
                # البحث في قاعدة المعرفة
                relevant_docs = []
                search_info = ""
                
                if knowledge_base and model:
                    relevant_docs = advanced_semantic_search(prompt, model, knowledge_base)
                    if relevant_docs:
                        search_info = f"🔍 **تم العثور على {len(relevant_docs)} مرجع ذي صلة**\n"
                        for i, doc in enumerate(relevant_docs[:3]):
                            search_info += f"📖 {doc['source']}\n"
                        search_info += "\n"
                        st.info(search_info.strip())
                
                # الحصول على الإجابة
                start_time = time.time()
                answer, source_info, answer_type = response_system.get_comprehensive_answer(prompt, relevant_docs)
                response_time = time.time() - start_time
                
                # تحديد نجاح الاستجابة
                is_successful = "خطأ" not in answer and "تعذر" not in answer
                if is_successful:
                    st.session_state.successful_responses += 1
                
                # تحديد أيقونة المصدر
                source_icons = {
                    "محلي": "🏠", "Gemini": "🧠", "DeepSeek": "🚀", 
                    "HuggingFace": "🤗", "احتياطي": "🔄"
                }
                source_icon = "🧠"
                for key, icon in source_icons.items():
                    if key in answer_type:
                        source_icon = icon
                        break
                
                # تنسيق الإجابة النهائية
                response_with_metadata = f"""{answer}

---
### 📋 **معلومات الاستجابة**
- {source_icon} **المصدر**: {source_info}
- ⚡ **النوع**: {answer_type}
- 🕐 **زمن الاستجابة**: {response_time:.2f} ثانية
- 📊 **جودة البحث**: {"ممتازة" if relevant_docs else "عامة"}

*💡 للحصول على معلومات أكثر تفصيلاً، جرب أسئلة محددة أكثر!*"""
                
                st.markdown(response_with_metadata)
                st.session_state.messages.append({"role": "assistant", "content": response_with_metadata})

    # تذييل الصفحة
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🧬 <strong>العرّاب للجينات v13.0</strong> - نظام ذكي متطور لخبرة وراثة الحمام</p>
        <p>⚡ مدعوم بتقنيات الذكاء الاصطناعي المتقدمة والبحث الدلالي الذكي</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
