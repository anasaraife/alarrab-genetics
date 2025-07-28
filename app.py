# ==============================================================================
#  العرّاب للجينات - الإصدار 12.1 المُحسَّن (مع إصلاح شامل للأخطاء)
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
from typing import List, Dict, Tuple
import time

# -------------------------------------------------
#  1. إعدادات الصفحة والمصادر
# -------------------------------------------------
st.set_page_config(
    page_title="العرّاب للجينات - الإصدار 12.1",
    page_icon="🧬",
    layout="wide",
)

BOOK_LINKS = [
    "https://drive.google.com/file/d/1CRwW78pd2RsKVd37elefz71RqwaCaute/view?usp=sharing",
    "https://drive.google.com/file/d/1894OOW1nEc3SkanLKKEzaXu_XhXYv8rF/view?usp=sharing",
]

# -------------------------------------------------
#  2. مدير نماذج الذكاء الاصطناعي المُحسَّن
# -------------------------------------------------
class AIModelManager:
    def __init__(self):
        self.models = {
            "gemini": {
                "name": "Google Gemini", 
                "available": self._check_gemini_key(), 
                "priority": 1,
                "status": "جاهز" if self._check_gemini_key() else "مفتاح API مفقود"
            },
            "deepseek": {
                "name": "DeepSeek", 
                "available": self._check_deepseek_key(), 
                "priority": 2,
                "status": "جاهز" if self._check_deepseek_key() else "مفتاح API مفقود"
            },
            "huggingface": {
                "name": "Hugging Face", 
                "available": self._check_huggingface_key(), 
                "priority": 3,
                "status": "جاهز" if self._check_huggingface_key() else "مفتاح API مفقود"
            },
            "fallback": {
                "name": "النمط الاحتياطي", 
                "available": True, 
                "priority": 4,
                "status": "دائماً متاح"
            }
        }

    def _check_gemini_key(self) -> bool:
        """فحص مفتاح Gemini مع رسائل تشخيصية"""
        try:
            key = st.secrets.get("GEMINI_API_KEY", "")
            if not key:
                st.sidebar.warning("⚠️ مفتاح GEMINI_API_KEY غير موجود في secrets")
                return False
            if len(key) < 20:
                st.sidebar.warning("⚠️ مفتاح Gemini يبدو غير صحيح")
                return False
            return True
        except Exception as e:
            st.sidebar.error(f"خطأ في فحص مفتاح Gemini: {e}")
            return False

    def _check_deepseek_key(self) -> bool:
        """فحص مفتاح DeepSeek مع رسائل تشخيصية"""
        try:
            key = st.secrets.get("DEEPSEEK_API_KEY", "")
            if not key:
                st.sidebar.info("💡 مفتاح DEEPSEEK_API_KEY غير موجود (اختياري)")
                return False
            return True
        except Exception as e:
            st.sidebar.error(f"خطأ في فحص مفتاح DeepSeek: {e}")
            return False

    def _check_huggingface_key(self) -> bool:
        """فحص مفتاح Hugging Face مع رسائل تشخيصية"""
        try:
            key = st.secrets.get("HUGGINGFACE_API_KEY", "")
            if not key:
                st.sidebar.info("💡 مفتاح HUGGINGFACE_API_KEY غير موجود (اختياري)")
                return False
            return True
        except Exception as e:
            st.sidebar.error(f"خطأ في فحص مفتاح Hugging Face: {e}")
            return False

    def get_available_models(self) -> List[str]:
        available = [model for model, config in self.models.items() if config["available"]]
        return sorted(available, key=lambda x: self.models[x]["priority"])

    def get_model_status(self) -> Dict:
        return {k: v["status"] for k, v in self.models.items()}

# -------------------------------------------------
#  3. بناء قاعدة المعرفة المُحسَّن
# -------------------------------------------------
@st.cache_resource
def load_embedding_model():
    """تحميل نموذج التضمين مع معالجة الأخطاء"""
    try:
        return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    except Exception as e:
        st.error(f"خطأ في تحميل نموذج التضمين: {e}")
        return None

@st.cache_data(ttl=86400, show_spinner=False)
def load_knowledge_base(_model):
    """بناء قاعدة المعرفة مع معالجة محسنة للأخطاء"""
    if _model is None:
        return None
        
    db_path = os.path.join(tempfile.gettempdir(), "text_knowledge_v12_1.db")
    
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS knowledge (source TEXT, content TEXT UNIQUE)")
        
        cursor.execute("SELECT COUNT(*) FROM knowledge")
        if cursor.fetchone()[0] == 0:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, link in enumerate(BOOK_LINKS):
                try:
                    status_text.text(f"تحميل الكتاب {i+1} من {len(BOOK_LINKS)}...")
                    progress_bar.progress((i) / len(BOOK_LINKS))
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                        file_id = link.split('/d/')[1].split('/')[0]
                        gdown.download(id=file_id, output=tmp.name, quiet=True)
                        
                        with open(tmp.name, 'rb') as f:
                            reader = PyPDF2.PdfReader(f)
                            for page_num, page in enumerate(reader.pages):
                                text = page.extract_text() or ""
                                if len(text.strip()) > 100:  # تقليل الحد الأدنى
                                    cursor.execute(
                                        "INSERT OR IGNORE INTO knowledge (source, content) VALUES (?, ?)",
                                        (f"الكتاب {i+1}، الصفحة {page_num+1}", text.strip())
                                    )
                        os.remove(tmp.name)
                        
                except Exception as e:
                    st.warning(f"تعذر تحميل الكتاب {i+1}: {e}")
                    continue
            
            conn.commit()
            progress_bar.progress(1.0)
            status_text.text("تم الانتهاء من بناء قاعدة المعرفة!")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()

        cursor.execute("SELECT source, content FROM knowledge")
        all_docs = [{"source": row[0], "content": row[1]} for row in cursor.fetchall()]
        conn.close()

        if not all_docs:
            st.warning("⚠️ قاعدة المعرفة فارغة")
            return None
        
        st.success(f"✅ تم تحميل {len(all_docs)} وثيقة في قاعدة المعرفة")
        
        contents = [doc['content'] for doc in all_docs]
        embeddings = _model.encode(contents, show_progress_bar=False)
        
        return {"documents": all_docs, "embeddings": embeddings}
        
    except Exception as e:
        st.error(f"خطأ في بناء قاعدة المعرفة: {e}")
        return None

# -------------------------------------------------
#  4. دوال البحث والذكاء الاصطناعي المُحسَّن
# -------------------------------------------------
def search_semantic_knowledge(query, model, knowledge_base, limit=5):
    """البحث الدلالي مع تحسينات"""
    if not knowledge_base or not model:
        return []
    
    try:
        query_embedding = model.encode([query])
        similarities = cosine_similarity(query_embedding, knowledge_base['embeddings'])[0]
        top_indices = np.argsort(similarities)[-limit:][::-1]
        
        # تقليل عتبة التشابه للحصول على نتائج أكثر
        results = [knowledge_base['documents'][i] for i in top_indices if similarities[i] > 0.25]
        
        return results
    except Exception as e:
        st.error(f"خطأ في البحث الدلالي: {e}")
        return []

class EnhancedAIResponder:
    def __init__(self, ai_manager: AIModelManager):
        self.ai_manager = ai_manager
        self.available_models = ai_manager.get_available_models()

    def get_gemini_response(self, query: str, context_docs: List[Dict]) -> Tuple[str, bool]:
        """الحصول على إجابة من Gemini مع معالجة محسنة للأخطاء"""
        try:
            API_KEY = st.secrets.get("GEMINI_API_KEY", "")
            if not API_KEY:
                return "مفتاح Gemini API غير موجود", False
                
            API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
            
            # إعداد السياق
            if context_docs:
                context = "\n\n".join([f"المصدر: {doc['source']}\nالمحتوى: {doc['content'][:800]}..." for doc in context_docs])
                prompt = f"""أنت خبير في وراثة الحمام. أجب على السؤال التالي بناءً على المعلومات المتوفرة:

السياق العلمي:
{context}

السؤال: {query}

الإجابة (بالعربية، واضحة ومفصلة):"""
            else:
                prompt = f"""أنت خبير في وراثة الحمام والطيور. أجب على هذا السؤال بالعربية بشكل علمي ودقيق:

{query}

الإجابة:"""

            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 1000,
                    "topP": 0.8,
                    "topK": 40
                }
            }
            
            response = requests.post(
                API_URL, 
                json=payload, 
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    answer = result['candidates'][0]['content']['parts'][0]['text']
                    return answer, True
                else:
                    return "لم يتم العثور على إجابة في الاستجابة", False
            else:
                return f"خطأ HTTP {response.status_code}: {response.text}", False
                
        except requests.exceptions.Timeout:
            return "انتهت مهلة الاتصال مع Gemini", False
        except requests.exceptions.RequestException as e:
            return f"خطأ في الاتصال: {str(e)}", False
        except Exception as e:
            return f"خطأ غير متوقع: {str(e)}", False

    def get_deepseek_response(self, query: str) -> Tuple[str, bool]:
        """الحصول على إجابة من DeepSeek"""
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
                        "content": "أنت خبير في وراثة الحمام والطيور. أجب بالعربية بطريقة علمية ودقيقة."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content']
                return answer, True
            else:
                return f"خطأ DeepSeek: {response.status_code}", False
                
    def get_huggingface_response(self, query: str) -> Tuple[str, bool]:
        """الحصول على إجابة من Hugging Face"""
        try:
            API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", "")
            if not API_KEY:
                return "مفتاح Hugging Face API غير موجود", False
            
            # استخدام نموذج مناسب للغة العربية
            API_URL = "https://api-inference.huggingface.co/models/aubmindlab/bert-base-arabertv2"
            headers = {"Authorization": f"Bearer {API_KEY}"}
            
            # تجربة نموذج للمحادثة
            API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
            
            payload = {
                "inputs": f"As a pigeon genetics expert, answer this question in Arabic: {query}",
                "parameters": {
                    "max_length": 500,
                    "temperature": 0.7,
                    "do_sample": True
                }
            }
            
            response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    answer = result[0].get('generated_text', 'لم أتمكن من توليد إجابة مناسبة.')
                    return answer, True
                else:
                    return "استجابة غير متوقعة من Hugging Face", False
            elif response.status_code == 503:
                return "نموذج Hugging Face قيد التحميل، يرجى المحاولة لاحقاً", False
            else:
                return f"خطأ Hugging Face: {response.status_code}", False
                
        except Exception as e:
            return f"خطأ في Hugging Face: {str(e)}", False

    def get_fallback_response(self, query: str) -> str:
        """إجابة احتياطية عندما تفشل جميع النماذج"""
        fallback_responses = {
            "color": "الألوان في الحمام موضوع معقد يحتاج لدراسة الجينات المسؤولة عن إنتاج الصبغات.",
            "genetics": "الوراثة في الحمام تتبع قوانين مندل الأساسية مع بعض التعقيدات الخاصة.",
            "breeding": "التربية الانتقائية تتطلب فهماً عميقاً للأنماط الوراثية والخصائص المرغوبة.",
            "mutation": "الطفرات في الحمام تؤثر على الشكل واللون وأحياناً السلوك.",
            "pigeon": "الحمام له تنوع وراثي هائل يظهر في الألوان والأشكال والأحجام المختلفة."
        }
        
        query_lower = query.lower()
        for keyword, response in fallback_responses.items():
            if keyword in query_lower or any(arabic_word in query for arabic_word in ["لون", "وراثة", "تربية", "طفرة", "حمام"]):
                return f"{response}\n\nللحصول على معلومات أكثر تفصيلاً، يرجى التأكد من إعداد مفاتيح API أو تحسين الاتصال بالإنترنت."
        
        return "عذراً، لا يمكنني الإجابة على هذا السؤال حالياً. يرجى التحقق من الاتصال بالإنترنت وإعدادات API."

    def get_comprehensive_answer(self, query: str, context_docs: List[Dict]) -> Tuple[str, str, str]:
        """الحصول على إجابة شاملة مع معالجة محسنة"""
        
        # أولاً: محاولة Gemini مع السياق المحلي
        if context_docs and "gemini" in self.available_models:
            answer, success = self.get_gemini_response(query, context_docs)
            if success:
                sources = ", ".join(list(set([doc['source'] for doc in context_docs[:3]])))
                return answer, f"قاعدة المعرفة المحلية: {sources}", "محلي + Gemini"
        
        # ثانياً: محاولة Gemini بدون سياق محلي
        if "gemini" in self.available_models:
            answer, success = self.get_gemini_response(query, [])
            if success:
                return answer, "Google Gemini (معرفة عامة)", "Gemini عام"
        
        # ثالثاً: محاولة DeepSeek
        if "deepseek" in self.available_models:
            answer, success = self.get_deepseek_response(query)
            if success:
                return answer, "DeepSeek AI", "DeepSeek"
        
        # رابعاً: محاولة Hugging Face
        if "huggingface" in self.available_models:
            answer, success = self.get_huggingface_response(query)
            if success:
                return answer, "Hugging Face AI", "HuggingFace"
        
        # أخيراً: الإجابة الاحتياطية
        fallback_answer = self.get_fallback_response(query)
        return fallback_answer, "النمط الاحتياطي", "احتياطي"

# -------------------------------------------------
#  5. واجهة المستخدم المُحسَّنة
# -------------------------------------------------
def main():
    st.title("🧬 العرّاب للجينات - الإصدار 12.1 المُحسَّن")
    st.markdown("### حاور خبير الوراثة الذكي مع تشخيص متقدم للأخطاء")

    # تحميل النماذج والأدوات
    model = load_embedding_model()
    ai_manager = AIModelManager()
    
    # الشريط الجانبي للتشخيص
    with st.sidebar:
        st.header("🔍 تشخيص النظام")
        
        # حالة النماذج
        st.subheader("حالة النماذج:")
        model_status = ai_manager.get_model_status()
        for model_name, status in model_status.items():
            if "جاهز" in status:
                st.success(f"✅ {ai_manager.models[model_name]['name']}: {status}")
            elif "مفقود" in status:
                st.error(f"❌ {ai_manager.models[model_name]['name']}: {status}")
            else:
                st.info(f"💡 {ai_manager.models[model_name]['name']}: {status}")
        
        # حالة قاعدة المعرفة
        st.subheader("قاعدة المعرفة:")
        if model:
            st.success("✅ نموذج التضمين جاهز")
        else:
            st.error("❌ فشل تحميل نموذج التضمين")
        
        # معلومات الجلسة
        if "query_count" not in st.session_state:
            st.session_state.query_count = 0
        
        st.subheader("إحصائيات:")
        st.metric("عدد الأسئلة", st.session_state.query_count)

    # تحميل قاعدة المعرفة
    knowledge_base = load_knowledge_base(model) if model else None
    ai_responder = EnhancedAIResponder(ai_manager)

    # إعداد المحادثة
    if "messages" not in st.session_state:
        welcome_msg = """أهلاً بك! أنا العرّاب الإصدار 12.1 المُحسَّن 🧬

**ما الجديد:**
- ✅ تشخيص متقدم للأخطاء
- ✅ معالجة محسنة لمفاتيح API
- ✅ نمط احتياطي للإجابة على الأسئلة
- ✅ واجهة تشخيصية في الشريط الجانبي

يمكنني الإجابة على أسئلتك حول وراثة الحمام حتى لو لم تكن جميع الخدمات متاحة!"""
        
        st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]

    # عرض المحادثة
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # معالجة الإدخال الجديد
    if prompt := st.chat_input("اسأل عن أي شيء متعلق بوراثة الحمام..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.query_count += 1
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("العرّاب يبحث ويفكر... 🤔"):
                # البحث في قاعدة المعرفة
                relevant_docs = []
                if knowledge_base and model:
                    relevant_docs = search_semantic_knowledge(prompt, model, knowledge_base)
                    if relevant_docs:
                        st.info(f"🔍 تم العثور على {len(relevant_docs)} وثيقة ذات صلة")
                
                # الحصول على الإجابة
                answer, source_info, answer_type = ai_responder.get_comprehensive_answer(prompt, relevant_docs)
                
                # تحديد لون المصدر
                if answer_type.startswith("محلي"):
                    source_color = "🏠"
                elif "Gemini" in answer_type:
                    source_color = "🧠"
                elif "DeepSeek" in answer_type:
                    source_color = "🚀"
                elif "HuggingFace" in answer_type:
                    source_color = "🤗"
                else:
                    source_color = "🔄"
                
                response_with_source = f"{answer}\n\n---\n*{source_color} المصدر: {source_info} ({answer_type})*"
                
                st.markdown(response_with_source)
                st.session_state.messages.append({"role": "assistant", "content": response_with_source})

if __name__ == "__main__":
    main()
