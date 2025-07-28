# ==============================================================================
#  العرّاب للجينات - الإصدار 11.0 المتقدم (مع AI متعدد النماذج)
# ==============================================================================

import streamlit as st
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import gdown
import PyPDF2
import os
import tempfile
import hashlib
import requests
import json
import numpy as np
from typing import List, Dict, Optional
import time
from datetime import datetime

# -------------------------------------------------
#  1. إعدادات الصفحة والمصادر
# -------------------------------------------------
st.set_page_config(
    page_title="العرّاب للجينات - الإصدار 11.0 المتقدم",
    page_icon="🧬",
    layout="wide",
)

# قائمة روابط الكتب
BOOK_LINKS = [
    "https://drive.google.com/file/d/1CRwW78pd2RsKVd37elefz71RqwaCaute/view?usp=sharing",
    "https://drive.google.com/file/d/1894OOW1nEc3SkanLKKEzaXu_XhXYv8rF/view?usp=sharing",
]

# -------------------------------------------------
#  2. إعدادات نماذج الذكاء الاصطناعي المتعددة
# -------------------------------------------------

class AIModelManager:
    """مدير لإدارة نماذج الذكاء الاصطناعي المتعددة"""
    
    def __init__(self):
        self.models = {
            "gemini": {
                "name": "Google Gemini",
                "available": self._check_gemini_availability(),
                "priority": 1
            },
            "deepseek": {
                "name": "DeepSeek",
                "available": self._check_deepseek_availability(),
                "priority": 2
            },
            "huggingface": {
                "name": "Hugging Face",
                "available": True,
                "priority": 3
            },
            "ollama": {
                "name": "Ollama Local",
                "available": self._check_ollama_availability(),
                "priority": 4
            }
        }
    
    def _check_gemini_availability(self) -> bool:
        """فحص توفر مفتاح Gemini API"""
        try:
            return "GEMINI_API_KEY" in st.secrets
        except:
            return False
    
    def _check_deepseek_availability(self) -> bool:
        """فحص توفر مفتاح DeepSeek API"""
        try:
            return "DEEPSEEK_API_KEY" in st.secrets
        except:
            return False
    
    def _check_ollama_availability(self) -> bool:
        """فحص توفر خدمة Ollama المحلية"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> List[str]:
        """الحصول على قائمة النماذج المتاحة مرتبة حسب الأولوية"""
        available = [model for model, config in self.models.items() if config["available"]]
        return sorted(available, key=lambda x: self.models[x]["priority"])

# -------------------------------------------------
#  3. تحميل النماذج وبناء قاعدة المعرفة الدلالية
# -------------------------------------------------

@st.cache_resource
def load_embedding_model():
    """تحميل نموذج الذكاء الاصطناعي متعدد اللغات."""
    return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

@st.cache_resource
def initialize_ai_manager():
    """تهيئة مدير نماذج الذكاء الاصطناعي"""
    return AIModelManager()

@st.cache_data(ttl=86400)
def load_knowledge_base(_model):
    """
    بناء قاعدة المعرفة (النصوص والمتجهات) من المصادر.
    """
    db_path = os.path.join(tempfile.gettempdir(), "text_knowledge_v11.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS knowledge (source TEXT, content TEXT UNIQUE)")
    
    cursor.execute("SELECT COUNT(*) FROM knowledge")
    if cursor.fetchone()[0] == 0:
        with st.spinner("يتم تحديث قاعدة المعرفة من المراجع العلمية..."):
            for i, link in enumerate(BOOK_LINKS):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                        file_id = link.split('/d/')[1].split('/')[0]
                        gdown.download(id=file_id, output=tmp.name, quiet=True)
                        with open(tmp.name, 'rb') as f:
                            reader = PyPDF2.PdfReader(f)
                            for page_num, page in enumerate(reader.pages):
                                text = page.extract_text() or ""
                                if len(text.strip()) > 150:
                                    cursor.execute("INSERT OR IGNORE INTO knowledge (source, content) VALUES (?, ?)",
                                                   (f"Book {i+1}, Page {page_num+1}", text.strip()))
                        os.remove(tmp.name)
                except Exception as e:
                    print(f"Could not process book {i+1}: {e}")
            conn.commit()

    cursor.execute("SELECT source, content FROM knowledge")
    all_docs = [{"source": row[0], "content": row[1]} for row in cursor.fetchall()]
    conn.close()

    if not all_docs:
        return None

    with st.spinner("جاري تحليل وفهرسة المعرفة..."):
        contents = [doc['content'] for doc in all_docs]
        embeddings = _model.encode(contents, show_progress_bar=True)
    
    return {"documents": all_docs, "embeddings": embeddings}

# -------------------------------------------------
#  4. دوال البحث والذكاء الاصطناعي المتقدم
# -------------------------------------------------

def search_semantic_knowledge(query, model, knowledge_base, limit=3):
    """البحث في قاعدة المعرفة باستخدام التشابه الدلالي."""
    if not knowledge_base:
        return []
    
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, knowledge_base['embeddings'])[0]
    top_indices = np.argsort(similarities)[-limit:][::-1]
    
    # رفع عتبة التشابه لضمان جودة النتائج
    return [knowledge_base['documents'][i] for i in top_indices if similarities[i] > 0.4]

class EnhancedAIResponder:
    """مجيب ذكي متقدم يستخدم عدة نماذج AI"""
    
    def __init__(self, ai_manager: AIModelManager):
        self.ai_manager = ai_manager
        self.available_models = ai_manager.get_available_models()
    
    def get_gemini_response(self, query: str, context_docs: List[Dict]) -> str:
        """الحصول على إجابة من Gemini"""
        try:
            API_KEY = st.secrets["GEMINI_API_KEY"]
            # استخدام نموذج Gemini 1.5 Flash للحصول على أداء أفضل مع التكلفة المنخفضة
            API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
            
            context = "\n\n".join([f"Source: {doc['source']}\nContent: {doc['content']}" for doc in context_docs])
            
            prompt = f"""
            أنت خبير عالمي في وراثة الحمام تُدعى 'العرّاب'.
            بناءً **حصرياً** على السياق العلمي المقدم أدناه، أجب على سؤال المستخدم بالعربية الواضحة والمحادثة.
            إذا لم تكن الإجابة موجودة في السياق، يجب أن تذكر أن المعلومة غير متاحة في الوثائق المقدمة.

            **السياق العلمي:**
            ---
            {context}
            ---

            **سؤال المستخدم:**
            {query}

            **إجابتك (بالعربية، بناءً على السياق فقط):**
            """
            
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            response = requests.post(API_URL, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            return f"خطأ في Gemini: {str(e)}"
    
    def get_deepseek_response(self, query: str) -> str:
        """الحصول على إجابة من DeepSeek (للأسئلة العامة)"""
        try:
            API_KEY = st.secrets["DEEPSEEK_API_KEY"]
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
            
            response = requests.post(API_URL, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            return f"خطأ في DeepSeek: {str(e)}"
    
    def get_huggingface_response(self, query: str) -> str:
        """الحصول على إجابة من Hugging Face (نموذج مجاني)"""
        try:
            # استخدام نموذج مجاني من Hugging Face
            API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
            
            # ملاحظة: يمكن استخدام نماذج أخرى مثل GPT-2 أو نماذج عربية
            headers = {"Authorization": f"Bearer {st.secrets.get('HUGGINGFACE_API_KEY', '')}"}
            
            payload = {
                "inputs": f"كخبير في وراثة الحمام، أجب على هذا السؤال: {query}",
                "parameters": {"max_length": 500, "temperature": 0.7}
            }
            
            response = requests.post(API_URL, json=payload, headers=headers)
            if response.status_code == 200:
                result = response.json()
                return result[0]['generated_text'] if isinstance(result, list) else "لم أتمكن من الحصول على إجابة واضحة."
            else:
                return "خدمة Hugging Face غير متاحة حالياً."
        except Exception as e:
            return f"خطأ في Hugging Face: {str(e)}"
    
    def get_ollama_response(self, query: str) -> str:
        """الحصول على إجابة من Ollama المحلي"""
        try:
            payload = {
                "model": "llama3.2",  # أو أي نموذج متاح محلياً
                "prompt": f"كخبير في وراثة الحمام والطيور، أجب على هذا السؤال بالعربية: {query}",
                "stream": False
            }
            
            response = requests.post("http://localhost:11434/api/generate", json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get('response', 'لم أتمكن من الحصول على إجابة.')
        except Exception as e:
            return f"خطأ في Ollama: {str(e)}"
    
    def get_comprehensive_answer(self, query: str, context_docs: List[Dict]) -> tuple:
        """الحصول على إجابة شاملة باستخدام عدة نماذج"""
        
        # أولاً: محاولة الإجابة بناءً على قاعدة المعرفة المحلية
        if context_docs:
            if "gemini" in self.available_models:
                local_answer = self.get_gemini_response(query, context_docs)
                sources = "، ".join(list(set([doc['source'] for doc in context_docs])))
                return local_answer, f"المصادر المحلية: {sources}", "محلي"
        
        # ثانياً: إذا لم تكن هناك نتائج مناسبة، استخدم النماذج الخارجية
        for model in self.available_models:
            try:
                if model == "deepseek":
                    answer = self.get_deepseek_response(query)
                    if not answer.startswith("خطأ"):
                        return answer, "DeepSeek AI", "خارجي"
                elif model == "huggingface":
                    answer = self.get_huggingface_response(query)
                    if not answer.startswith("خطأ"):
                        return answer, "Hugging Face AI", "خارجي"
                elif model == "ollama":
                    answer = self.get_ollama_response(query)
                    if not answer.startswith("خطأ"):
                        return answer, "Ollama المحلي", "محلي متقدم"
            except Exception as e:
                continue
        
        return "عذراً، لم أتمكن من الحصول على إجابة مناسبة لسؤالك من أي من النماذج المتاحة.", "غير متاح", "فشل"

# -------------------------------------------------
#  5. واجهة المستخدم المحسّنة
# -------------------------------------------------

def main():
    st.title("🧬 العرّاب للجينات - الإصدار 11.0 المتقدم")
    st.markdown("### حاور خبير الوراثة الذكي مع قدرات AI متقدمة متعددة النماذج")
    
    # تحميل النماذج والأدوات
    model = load_embedding_model()
    ai_manager = initialize_ai_manager()
    knowledge_base = load_knowledge_base(model)
    ai_responder = EnhancedAIResponder(ai_manager)
    
    # عرض النماذج المتاحة في الشريط الجانبي
    with st.sidebar:
        st.header("🤖 النماذج المتاحة")
        available_models = ai_manager.get_available_models()
        
        if available_models:
            for model_key in available_models:
                model_info = ai_manager.models[model_key]
                st.success(f"✅ {model_info['name']}")
        else:
            st.warning("⚠️ لا توجد نماذج متاحة")
        
        st.header("📊 إحصائيات الجلسة")
        if "query_count" not in st.session_state:
            st.session_state.query_count = 0
        if "local_answers" not in st.session_state:
            st.session_state.local_answers = 0
        if "external_answers" not in st.session_state:
            st.session_state.external_answers = 0
        
        st.metric("إجمالي الأسئلة", st.session_state.query_count)
        st.metric("إجابات محلية", st.session_state.local_answers)
        st.metric("إجابات خارجية", st.session_state.external_answers)
    
    # إعداد المحادثة
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "أهلاً بك! أنا العرّاب الذكي الإصدار 11.0. أستطيع الإجابة على أسئلتك من مصادري المحلية، وإذا لم أجد الإجابة، سأتواصل مع نماذج ذكاء اصطناعي أخرى للحصول على أفضل إجابة ممكنة. اسأل عن أي شيء متعلق بوراثة الحمام!"
            }
        ]
    
    # عرض المحادثة
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # معالجة الإدخال الجديد
    if prompt := st.chat_input("اسأل عن جين، طفرة، أو نمط وراثي..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            if not available_models:
                st.error("لا توجد نماذج ذكاء اصطناعي متاحة حالياً. يرجى التحقق من الإعدادات.")
            else:
                with st.spinner("العرّاب يفكر ويبحث في مصادر متعددة..."):
                    # البحث في قاعدة المعرفة المحلية
                    relevant_docs = []
                    if knowledge_base:
                        relevant_docs = search_semantic_knowledge(prompt, model, knowledge_base)
                    
                    # الحصول على إجابة شاملة
                    answer, source_info, answer_type = ai_responder.get_comprehensive_answer(prompt, relevant_docs)
                    
                    # تحديث الإحصائيات
                    st.session_state.query_count += 1
                    if answer_type == "محلي":
                        st.session_state.local_answers += 1
                    elif answer_type in ["خارجي", "محلي متقدم"]:
                        st.session_state.external_answers += 1
                    
                    # عرض الإجابة مع معلومات المصدر
                    response_with_source = f"{answer}\n\n*المصدر: {source_info}*"
                    
                    # إضافة مؤشر نوع الإجابة
                    if answer_type == "محلي":
                        response_with_source += "\n\n🏠 *تم الحصول على هذه الإجابة من قاعدة المعرفة المحلية*"
                    elif answer_type == "خارجي":
                        response_with_source += "\n\n🌐 *تم الحصول على هذه الإجابة من نموذج ذكاء اصطناعي خارجي*"
                    elif answer_type == "محلي متقدم":
                        response_with_source += "\n\n🖥️ *تم الحصول على هذه الإجابة من نموذج محلي متقدم*"
                
                st.markdown(response_with_source)
                st.session_state.messages.append({"role": "assistant", "content": response_with_source})

if __name__ == "__main__":
    main()
