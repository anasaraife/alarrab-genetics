# ==============================================================================
#  العرّاب للجينات - الإصدار 12.0 المتقدم (المرن والمتعدد المصادر)
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
from typing import List, Dict

# -------------------------------------------------
#  1. إعدادات الصفحة والمصادر
# -------------------------------------------------
st.set_page_config(
    page_title="العرّاب للجينات - الإصدار 12.0",
    page_icon="🧬",
    layout="wide",
)

BOOK_LINKS = [
    "https://drive.google.com/file/d/1CRwW78pd2RsKVd37elefz71RqwaCaute/view?usp=sharing",
    "https://drive.google.com/file/d/1894OOW1nEc3SkanLKKEzaXu_XhXYv8rF/view?usp=sharing",
]

# -------------------------------------------------
#  2. مدير نماذج الذكاء الاصطناعي
# -------------------------------------------------
class AIModelManager:
    def __init__(self):
        self.models = {
            "gemini": {"name": "Google Gemini", "available": self._check_secret("GEMINI_API_KEY"), "priority": 1},
            "deepseek": {"name": "DeepSeek", "available": self._check_secret("DEEPSEEK_API_KEY"), "priority": 2},
        }

    def _check_secret(self, key: str) -> bool:
        try:
            return st.secrets.get(key) is not None and st.secrets[key] != ""
        except Exception:
            return False

    def get_available_models(self) -> List[str]:
        available = [model for model, config in self.models.items() if config["available"]]
        return sorted(available, key=lambda x: self.models[x]["priority"])

# -------------------------------------------------
#  3. بناء قاعدة المعرفة
# -------------------------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

@st.cache_data(ttl=86400)
def load_knowledge_base(_model):
    # This function builds the knowledge base from PDFs
    db_path = os.path.join(tempfile.gettempdir(), "text_knowledge_v12.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS knowledge (source TEXT, content TEXT UNIQUE)")
    
    cursor.execute("SELECT COUNT(*) FROM knowledge")
    if cursor.fetchone()[0] == 0:
        with st.spinner("تحديث قاعدة المعرفة..."):
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

    if not all_docs: return None
    
    contents = [doc['content'] for doc in all_docs]
    embeddings = _model.encode(contents, show_progress_bar=False)
    return {"documents": all_docs, "embeddings": embeddings}

# -------------------------------------------------
#  4. دوال البحث والذكاء الاصطناعي المتقدم
# -------------------------------------------------
def search_semantic_knowledge(query, model, knowledge_base, limit=3):
    if not knowledge_base: return []
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, knowledge_base['embeddings'])[0]
    top_indices = np.argsort(similarities)[-limit:][::-1]
    return [knowledge_base['documents'][i] for i in top_indices if similarities[i] > 0.4]

class EnhancedAIResponder:
    def __init__(self, ai_manager: AIModelManager):
        self.ai_manager = ai_manager
        self.available_models = ai_manager.get_available_models()

    def get_gemini_response(self, query: str, context_docs: List[Dict]) -> str:
        try:
            API_KEY = st.secrets["GEMINI_API_KEY"]
            API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
            context = "\n\n".join([f"Source: {doc['source']}\nContent: {doc['content']}" for doc in context_docs])
            prompt = f"Based ONLY on the context below, answer the user's question in Arabic.\n\nContext:\n{context}\n\nUser Question: {query}\n\nAnswer (in Arabic):"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            response = requests.post(API_URL, json=payload, timeout=20)
            response.raise_for_status()
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except KeyError:
            return "ERROR: مفتاح Gemini API غير موجود في الإعدادات."
        except requests.exceptions.RequestException as e:
            return f"ERROR: خطأ في الاتصال بخدمة Gemini: {e}"
        except Exception as e:
            return f"ERROR: خطأ غير متوقع في Gemini: {e}"

    def get_deepseek_response(self, query: str) -> str:
        try:
            API_KEY = st.secrets["DEEPSEEK_API_KEY"]
            API_URL = "https://api.deepseek.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": "deepseek-chat",
                "messages": [{"role": "system", "content": "You are an expert in pigeon genetics. Answer in Arabic."}, {"role": "user", "content": query}]
            }
            response = requests.post(API_URL, json=payload, headers=headers, timeout=20)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except KeyError:
            return "ERROR: مفتاح DeepSeek API غير موجود في الإعدادات."
        except Exception as e:
            return f"ERROR: خطأ في DeepSeek: {e}"

    def get_comprehensive_answer(self, query: str, context_docs: List[Dict]) -> tuple:
        # الأولوية الأولى: الإجابة من المصادر باستخدام Gemini
        if context_docs and "gemini" in self.available_models:
            answer = self.get_gemini_response(query, context_docs)
            if not answer.startswith("ERROR:"):
                sources = ", ".join(list(set([doc['source'] for doc in context_docs])))
                return answer, f"المصادر المحلية: {sources}", "محلي (Gemini RAG)"

        # إذا فشل الخيار الأول، ننتقل للنماذج الخارجية كخطة بديلة
        for model_key in self.available_models:
            answer = ""
            source_name = ""
            if model_key == "deepseek":
                answer = self.get_deepseek_response(query)
                source_name = "DeepSeek AI"
            
            if answer and not answer.startswith("ERROR:"):
                return answer, source_name, "خارجي"
        
        return "عذراً، لم أتمكن من الحصول على إجابة. جميع النماذج المتاحة تواجه مشاكل حالياً.", "N/A", "فشل"

# -------------------------------------------------
#  5. واجهة المستخدم
# -------------------------------------------------
st.title("🧬 العرّاب للجينات - الإصدار 12.0")
st.markdown("### حاور خبير الوراثة الذكي مع قدرات AI مرنة ومتعددة المصادر")

model = load_embedding_model()
ai_manager = AIModelManager()
knowledge_base = load_knowledge_base(model)
ai_responder = EnhancedAIResponder(ai_manager)

with st.sidebar:
    st.header("🤖 حالة النماذج")
    for model_key, config in ai_manager.models.items():
        st.write(f"{config['name']}: {'✅ متاح' if config['available'] else '❌ غير متاح'}")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "أهلاً بك! أنا العرّاب الإصدار 12.0. كيف يمكنني مساعدتك؟"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("اسأل عن أي شيء متعلق بوراثة الحمام..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("العرّاب يفكر ويتشاور مع الخبراء..."):
            relevant_docs = search_semantic_knowledge(prompt, model, knowledge_base)
            answer, source_info, answer_type = ai_responder.get_comprehensive_answer(prompt, relevant_docs)
            
            response_with_source = f"{answer}\n\n*المصدر: {source_info} ({answer_type})*"
            st.markdown(response_with_source)
            st.session_state.messages.append({"role": "assistant", "content": response_with_source})
