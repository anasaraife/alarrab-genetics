# ==============================================================================
#  العرّاب للجينات - الإصدار 9.0 (الذكاء الموثق بالمصادر - RAG)
#  - يبحث في المصادر المحلية أولاً ثم يستخدم Gemini لصياغة إجابة موثقة.
# ==============================================================================

import streamlit as st
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gdown
import PyPDF2
import os
import tempfile
import hashlib
import requests
import json

# -------------------------------------------------
#  1. إعدادات الصفحة والمصادر
# -------------------------------------------------
st.set_page_config(
    page_title="العرّاب للجينات - الإصدار 9.0",
    page_icon="🕊️",
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
#  2. إعداد قاعدة البيانات المحلية (SQLite)
# -------------------------------------------------

@st.cache_resource
def init_sqlite_db():
    """إنشاء قاعدة بيانات SQLite في مجلد مؤقت."""
    db_path = os.path.join(tempfile.gettempdir(), "genetics_knowledge_v9.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id INTEGER PRIMARY KEY, content TEXT NOT NULL, source TEXT NOT NULL, content_hash TEXT UNIQUE
        )
    """)
    conn.commit()
    return conn

@st.cache_data(ttl=86400) # تخزين مؤقت ليوم واحد
def build_knowledge_base_from_sources(_conn):
    """بناء قاعدة المعرفة من الكتب إذا كانت فارغة."""
    cursor = _conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM knowledge_base")
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
                                    content_hash = hashlib.md5(text.encode()).hexdigest()
                                    cursor.execute("INSERT OR IGNORE INTO knowledge_base (content, source, content_hash) VALUES (?, ?, ?)",
                                                   (text.strip(), f"Book {i+1}, Page {page_num+1}", content_hash))
                        os.remove(tmp.name)
                except Exception as e:
                    # تجاهل الكتب التي تفشل في التحميل بصمت لتجنب إيقاف التطبيق
                    print(f"Could not process book {i+1}: {e}")
                    pass
            _conn.commit()
    return True

# -------------------------------------------------
#  3. دوال البحث والذكاء الاصطناعي (RAG)
# -------------------------------------------------

def search_local_knowledge(query, conn, limit=3):
    """البحث في قاعدة البيانات المحلية باستخدام TF-IDF للعثور على السياق."""
    cursor = conn.cursor()
    cursor.execute("SELECT content, source FROM knowledge_base")
    results = cursor.fetchall()
    if not results: return []
    
    documents = [row[0] for row in results]
    sources = [row[1] for row in results]
    
    try:
        vectorizer = TfidfVectorizer(max_features=1000).fit(documents)
        tfidf_matrix = vectorizer.transform(documents)
        query_vector = vectorizer.transform([query])
        
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-limit:][::-1]
        
        return [{"content": documents[i], "source": sources[i]} for i in top_indices if similarities[i] > 0.1]
    except ValueError:
        # يحدث هذا الخطأ إذا كان قاموس المفردات فارغًا (لا توجد مستندات)
        return []


@st.cache_data
def get_rag_answer_with_gemini(query, context_docs):
    """
    يستخدم Gemini API لصياغة إجابة ذكية بناءً على السياق المسترجع من المصادر.
    """
    API_KEY = ""
    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
    
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
        response = requests.post(API_URL, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"حدث خطأ أثناء التواصل مع الوكيل الذكي. يرجى المحاولة مرة أخرى. الخطأ: {str(e)}"

# -------------------------------------------------
#  4. واجهة المستخدم
# -------------------------------------------------

st.title("🕊️ العرّاب للجينات - الإصدار 9.0 (الذكي والموثق)")
st.markdown("حاور خبير الوراثة للحصول على إجابات دقيقة من المراجع العلمية المعتمدة")

# تهيئة النظام
db_conn = init_sqlite_db()
build_knowledge_base_from_sources(db_conn)

# إعداد ذاكرة الجلسة للمحادثة
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "أهلاً بك! أنا العرّاب. اسألني أي سؤال، وسأبحث لك عن الإجابة في المراجع العلمية."}]

# عرض رسائل المحادثة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# حقل إدخال المستخدم
if prompt := st.chat_input("اسأل عن جين، طفرة، أو نمط وراثي..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("العرّاب يبحث في المراجع ويفكر..."):
            # 1. البحث في المصادر المحلية للعثور على السياق
            relevant_docs = search_local_knowledge(prompt, db_conn)
            
            if not relevant_docs:
                response = "لم أتمكن من العثور على معلومات ذات صلة بسؤالك في قاعدة المعرفة الحالية."
            else:
                # 2. توليد إجابة ذكية وموثقة باستخدام السياق
                response = get_rag_answer_with_gemini(prompt, relevant_docs)
                
                # إضافة المصادر إلى الإجابة
                sources = "، ".join(list(set([doc['source'] for doc in relevant_docs])))
                response += f"\n\n*المصادر التي تم الاعتماد عليها: {sources}*"
        
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
