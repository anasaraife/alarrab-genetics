# ==============================================================================
#  العرّاب للجينات - الإصدار 17.0 (الذكاء الموثق بالمصادر - RAG)
#  - يبحث في المصادر المحلية أولاً ثم يستخدم Gemini لصياغة إجابة موثقة.
# ==============================================================================

# --- HOT-PATCH FOR SQLITE3 ON STREAMLIT CLOUD ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --------------------------------------------------

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

# -------------------------------------------------
#  1. إعدادات الصفحة والمصادر
# -------------------------------------------------
st.set_page_config(
    page_title="العرّاب للجينات - الإصدار 17.0",
    page_icon="🧬",
    layout="wide",
)

# قائمة روابط الكتب
BOOK_LINKS = [
    "https://drive.google.com/file/d/1CRwW78pd2RsKVd37elefz71RqwaCaute/view?usp=sharing",
    "https://drive.google.com/file/d/1894OOW1nEc3SkanLKKEzaXu_XhXYv8rF/view?usp=sharing",
    "https://drive.google.com/file/d/18pc9PptjfcjQfPyVCiaSq30RFs3ZjXF4/view?usp=sharing",
    "https://drive.google.com/file/d/17hklyXm2R6ChYRddDbYRkqrtD8mE_nC_/view?usp=sharing",
    "https://drive.google.com/file/d/1Mq3zgz4NDm6guelOzuni3O4_2kaQpJAi/view?usp=sharing",
    "https://drive.google.com/file/d/1hoCxIPU9xJgsl1J-AnEG2E0AX3H5c5Kg/view?usp=sharing",
    "https://drive.google.com/file/d/14qInRfBTOhOJYsjs6tYRxAq1xFDrD-_O/view?usp=sharing",
    "https://drive.google.com/file/d/1kaVob_EdCP5v_H71nUS3O1-YairROV1b/view?usp=sharing"
]

# -------------------------------------------------
#  2. بناء قاعدة المعرفة الدلالية
# -------------------------------------------------

@st.cache_resource
def load_embedding_model():
    """تحميل نموذج الذكاء الاصطناعي متعدد اللغات."""
    return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

@st.cache_data(ttl=86400)
def load_knowledge_base(_model):
    """بناء قاعدة المعرفة (النصوص والمتجهات) من المصادر."""
    db_path = os.path.join(tempfile.gettempdir(), "text_knowledge_v17.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS knowledge (source TEXT, content TEXT UNIQUE)")
    
    cursor.execute("SELECT COUNT(*) FROM knowledge")
    if cursor.fetchone()[0] == 0:
        with st.spinner("يتم تحديث قاعدة المعرفة من المراجع العلمية لأول مرة..."):
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

    with st.spinner("جاري تحليل وفهرسة المعرفة..."):
        contents = [doc['content'] for doc in all_docs]
        embeddings = _model.encode(contents, show_progress_bar=True)
    
    return {"documents": all_docs, "embeddings": embeddings}

# -------------------------------------------------
#  3. دوال البحث والذكاء الاصطناعي (RAG)
# -------------------------------------------------

def search_semantic_knowledge(query, model, knowledge_base, limit=3):
    """البحث في قاعدة المعرفة باستخدام التشابه الدلالي."""
    if not knowledge_base: return []
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, knowledge_base['embeddings'])[0]
    top_indices = np.argsort(similarities)[-limit:][::-1]
    return [knowledge_base['documents'][i] for i in top_indices if similarities[i] > 0.3]

@st.cache_data
def get_rag_answer_with_gemini(query, context_docs):
    """يستخدم Gemini API لصياغة إجابة ذكية بناءً على السياق."""
    try:
        API_KEY = st.secrets["GEMINI_API_KEY"]
    except (FileNotFoundError, KeyError):
        return "خطأ: لم يتم العثور على مفتاح GEMINI_API_KEY في إعدادات التطبيق. يرجى التأكد من إضافته بشكل صحيح."

    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
    context = "\n\n".join([f"Source: {doc['source']}\nContent: {doc['content']}" for doc in context_docs])
    
    prompt = f"""
    You are a world-class expert in pigeon genetics named 'Al-Arrab'.
    Based **exclusively** on the scientific context provided below, answer the user's question in clear, conversational Arabic.
    If the answer is not in the context, you MUST state that the information is not available in the provided documents. Do not use any prior knowledge.

    **Scientific Context:**
    ---
    {context}
    ---

    **User's Question:**
    {query}

    **Your Answer (in Arabic, based only on the context):**
    """
    
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(API_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    except requests.exceptions.HTTPError as e:
        return f"حدث خطأ أثناء الاتصال بالوكيل الذكي (HTTP Error: {e.response.status_code}). تأكد من صحة مفتاح API الخاص بك."
    except Exception as e:
        return f"حدث خطأ أثناء التواصل مع الوكيل الذكي: {str(e)}"

# -------------------------------------------------
#  4. واجهة المستخدم
# -------------------------------------------------

st.title("🕊️ العرّاب للجينات - الإصدار 17.0 (الذكي والموثق)")
st.markdown("حاور خبير الوراثة للحصول على إجابات دقيقة من المراجع العلمية المعتمدة")

model = load_embedding_model()
knowledge_base = load_knowledge_base(model)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "أهلاً بك! أنا العرّاب. اسألني أي سؤال، وسأبحث لك عن الإجابة في المراجع العلمية الموثوقة."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("اسأل عن جين، طفرة، أو نمط وراثي..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not knowledge_base:
            st.error("قاعدة المعرفة غير متاحة حاليًا. قد تكون عملية البناء الأولية لا تزال جارية.")
        else:
            with st.spinner("العرّاب يبحث في المراجع ويفكر..."):
                relevant_docs = search_semantic_knowledge(prompt, model, knowledge_base)
                
                if not relevant_docs:
                    response = "لم أتمكن من العثور على معلومات ذات صلة بسؤالك في قاعدة المعرفة الحالية."
                else:
                    response = get_rag_answer_with_gemini(prompt, relevant_docs)
                    sources = "، ".join(list(set([doc['source'] for doc in relevant_docs])))
                    response += f"\n\n*المصادر التي تم الاعتماد عليها: {sources}*"
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
