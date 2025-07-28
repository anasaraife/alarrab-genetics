# ==============================================================================
#  مشروع العرّاب للجينات - الإصدار المحسن
# ==============================================================================

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import gdown
import PyPDF2
import os
import tempfile

# إعدادات الصفحة
st.set_page_config(
    page_title="العرّاب للجينات",
    page_icon="🕊️",
    layout="wide",
)

# روابط الكتب
BOOK_LINKS = [
    "https://drive.google.com/file/d/1CRwW78pd2RsKVd37elefz71RqwaCaute/view?usp=sharing",
    # ... باقي الروابط
]

@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    except Exception as e:
        st.error(f"فشل تحميل نموذج التضمين: {e}")
        st.stop()

@st.cache_resource
def init_chroma_db(_model):
    try:
        client = chromadb.PersistentClient(path=os.path.join(tempfile.gettempdir(), "chroma_db"))
        collection = client.get_or_create_collection(name="pigeon_genetics")
        
        if collection.count() == 0:
            with st.status("⚙️ جاري إعداد قاعدة المعرفة...") as status:
                load_documents(collection)
        
        return collection
    except Exception as e:
        st.error(f"فشل تهيئة قاعدة البيانات: {e}")
        st.stop()

@st.cache_data
def load_documents(collection):
    all_texts = []
    
    for link in BOOK_LINKS:
        try:
            with tempfile.NamedTemporaryFile(delete=True, suffix='.pdf') as tmp:
                file_id = link.split('/d/')[1].split('/')[0]
                gdown.download(id=file_id, output=tmp.name, quiet=True)
                
                text = extract_text_from_pdf(tmp.name)
                if text:
                    all_texts.append({'source': link, 'content': text})
        except Exception as e:
            st.warning(f"تخطي الكتاب {link}: {e}")
    
    if all_texts:
        process_and_add_texts(all_texts, collection)

def extract_text_from_pdf(filepath):
    text = ""
    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            if reader.is_encrypted:
                reader.decrypt("")
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
    except Exception as e:
        st.warning(f"خطأ في قراءة PDF: {e}")
    return text

def process_and_add_texts(texts, collection):
    chunks, metadatas, ids = [], [], []
    
    for i, doc in enumerate(texts):
        for chunk in doc['content'].split('\n\n'):
            if len(chunk.strip()) > 100:
                chunks.append(chunk.strip())
                metadatas.append({'source': doc['source']})
                ids.append(f"doc_{i}_{len(chunks)}")
    
    if chunks:
        collection.add(documents=chunks, metadatas=metadatas, ids=ids)

# واجهة المستخدم
st.title("🕊️ العرّاب للجينات - النسخة المحسنة")
st.write("ابحث في مراجع وراثة الحمام")

model = load_embedding_model()
db = init_chroma_db(model)

query = st.text_input("اكتب سؤالك هنا:")
if query:
    results = db.query(query_texts=[query], n_results=3)
    
    if results['documents']:
        for i, doc in enumerate(results['documents'][0]):
            similarity = (1 - results['distances'][0][i]) * 100
            source = results['metadatas'][0][i]['source']
            
            if i == 0:
                st.success(f"🔍 أفضل نتيجة ({similarity:.0f}% تطابق):")
                st.markdown(f"> {doc}")
                st.caption(f"المصدر: {source}")
            else:
                with st.expander(f"نتيجة إضافية ({similarity:.0f}%)"):
                    st.info(doc)
                    st.caption(f"المصدر: {source}")
    else:
        st.warning("لم يتم العثور على نتائج مطابقة")
