# ==============================================================================
#  العرّاب للجينات - الإصدار 18.0 (الخبير الديناميكي)
#  - يدمج قاعدة المعرفة الدائمة مع القدرة على تحليل الملفات المرفوعة ديناميكيًا.
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
    page_title="العرّاب للجينات - الإصدار 18.0",
    page_icon="🧬",
    layout="wide",
)

# قائمة روابط الكتب للمعرفة الدائمة
BOOK_LINKS = [
    "https://drive.google.com/file/d/1CRwW78pd2RsKVd37elefz71RqwaCaute/view?usp=sharing",
    "https://drive.google.com/file/d/1894OOW1nEc3SkanLKKEzaXu_XhXYv8rF/view?usp=sharing",
]

# -------------------------------------------------
#  2. بناء قواعد المعرفة (الدائمة والمؤقتة)
# -------------------------------------------------

@st.cache_resource
def load_embedding_model():
    """تحميل نموذج الذكاء الاصطناعي متعدد اللغات."""
    return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

@st.cache_data(ttl=86400)
def load_permanent_knowledge_base(_model):
    """بناء قاعدة المعرفة الدائمة من الكتب."""
    db_path = os.path.join(tempfile.gettempdir(), "permanent_knowledge_v18.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS knowledge (source TEXT, content TEXT UNIQUE)")
    
    cursor.execute("SELECT COUNT(*) FROM knowledge")
    if cursor.fetchone()[0] == 0:
        with st.spinner("تحديث قاعدة المعرفة الدائمة من المراجع..."):
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

def process_uploaded_file(uploaded_file, model):
    """معالجة الملف المرفوع وإنشاء قاعدة معرفة مؤقتة له."""
    if uploaded_file is None:
        return None
    
    with st.spinner(f"جاري تحليل ملف '{uploaded_file.name}'..."):
        try:
            text = ""
            if uploaded_file.type == "application/pdf":
                reader = PyPDF2.PdfReader(uploaded_file)
                for page in reader.pages:
                    text += (page.extract_text() or "") + "\n"
            else: # for .txt files
                text = uploaded_file.getvalue().decode("utf-8")

            chunks = [chunk.strip() for chunk in text.split('\n\n') if len(chunk.strip()) > 100]
            if not chunks:
                st.warning("لم يتم العثور على محتوى كافٍ في الملف لتحليله.")
                return None

            embeddings = model.encode(chunks, show_progress_bar=False)
            documents = [{"source": f"الملف المرفوع: {uploaded_file.name}", "content": chunk} for chunk in chunks]
            
            st.success(f"تم تحليل الملف بنجاح. يمكنك الآن طرح أسئلة حوله.")
            return {"documents": documents, "embeddings": embeddings}
        except Exception as e:
            st.error(f"فشل تحليل الملف: {e}")
            return None

# -------------------------------------------------
#  3. دوال البحث والذكاء الاصطناعي (RAG)
# -------------------------------------------------

def search_knowledge_base(query, model, knowledge_base, limit=2):
    """البحث في أي قاعدة معرفة (دائمة أو مؤقتة)."""
    if not knowledge_base: return []
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, knowledge_base['embeddings'])[0]
    top_indices = np.argsort(similarities)[-limit:][::-1]
    return [knowledge_base['documents'][i] for i in top_indices if similarities[i] > 0.3]

@st.cache_data
def get_rag_answer_with_gemini(query, context_docs):
    try:
        API_KEY = st.secrets["GEMINI_API_KEY"]
    except (FileNotFoundError, KeyError):
        return "خطأ: لم يتم العثور على مفتاح GEMINI_API_KEY."

    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
    context = "\n\n".join([f"Source: {doc['source']}\nContent: {doc['content']}" for doc in context_docs])
    prompt = f"Based exclusively on the scientific context provided, answer the user's question in Arabic.\n\nContext:\n---\n{context}\n---\n\nUser's Question:\n{query}\n\nAnswer (in Arabic):"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(API_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"حدث خطأ أثناء التواصل مع الوكيل الذكي: {str(e)}"

# -------------------------------------------------
#  4. واجهة المستخدم (تم إعادة الترتيب)
# -------------------------------------------------

# --- تحميل المكونات الأساسية أولاً ---
model = load_embedding_model()
permanent_kb = load_permanent_knowledge_base(model)

# --- الشريط الجانبي (بوابة المعرفة) ---
with st.sidebar:
    st.header("🧠 بوابة المعرفة")
    st.write("قم بتوسيع معرفة العرّاب بشكل مؤقت عن طريق رفع ملفاتك الخاصة.")
    
    uploaded_file = st.file_uploader(
        "ارفع ملف (PDF, TXT)",
        type=['pdf', 'txt'],
        help="سيتم تحليل الملف ويمكنك طرح أسئلة حوله خلال هذه الجلسة فقط."
    )
    
    if uploaded_file:
        st.session_state.temporary_kb = process_uploaded_file(uploaded_file, model)
    
    if st.button("🗑️ إزالة الملف المؤقت"):
        st.session_state.temporary_kb = None
        st.success("تمت إزالة ذاكرة الملف المؤقت.")
        st.rerun()

# --- واجهة المحادثة الرئيسية ---
st.title("🕊️ العرّاب للجينات - الإصدار 18.0 (الخبير الديناميكي)")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "أهلاً بك! أنا العرّاب. يمكنك سؤالي من معرفتي الدائمة، أو رفع ملف من الشريط الجانبي لمناقشته."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("اسأل عن جين، أو عن محتوى الملف المرفوع..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("العرّاب يبحث في جميع مصادره ويفكر..."):
            # البحث في كلا قاعدتي المعرفة
            # *** تم تصحيح اسم الدالة هنا ***
            permanent_docs = search_knowledge_base(prompt, model, permanent_kb)
            temporary_docs = []
            if 'temporary_kb' in st.session_state and st.session_state.temporary_kb:
                # *** تم تصحيح اسم الدالة هنا ***
                temporary_docs = search_knowledge_base(prompt, model, st.session_state.temporary_kb)

            combined_docs = permanent_docs + temporary_docs
            
            if not combined_docs:
                response = "لم أتمكن من العثور على معلومات ذات صلة بسؤالك في أي من المصادر المتاحة."
            else:
                response = get_rag_answer_with_gemini(prompt, combined_docs)
                sources = "، ".join(list(set([doc['source'] for doc in combined_docs])))
                response += f"\n\n*المصادر التي تم الاعتماد عليها: {sources}*"
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
