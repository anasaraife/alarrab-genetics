# ==============================================================================
#  HOT-PATCH FOR SQLITE3 VERSION ON STREAMLIT CLOUD
#  This is a workaround for the issue where Streamlit's default sqlite3
#  version is too old for ChromaDB. This code must run BEFORE chromadb is imported.
# ==============================================================================
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ==============================================================================


# ==============================================================================
#  مشروع العرّاب للجينات - الإصدار 3.1 (مع حل مشكلة التوافق)
# ==============================================================================

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import gdown
import PyPDF2
import os
import tempfile

# -------------------------------------------------
#  1. إعدادات الصفحة والمصادر
# -------------------------------------------------
st.set_page_config(
    page_title="العرّاب للجينات",
    page_icon="🕊️",
    layout="wide",
)

# قائمة روابط الكتب (سنعالج أول كتابين فقط في البداية)
BOOK_LINKS = [
    "[https://drive.google.com/file/d/1CRwW78pd2RsKVd37elefz71RqwaCaute/view?usp=sharing](https://drive.google.com/file/d/1CRwW78pd2RsKVd37elefz71RqwaCaute/view?usp=sharing)",
    "[https://drive.google.com/file/d/1894OOW1nEc3SkanLKKEzaXu_XhXYv8rF/view?usp=sharing](https://drive.google.com/file/d/1894OOW1nEc3SkanLKKEzaXu_XhXYv8rF/view?usp=sharing)",
    # "[https://drive.google.com/file/d/18pc9PptjfcjQfPyVCiaSq30RFs3ZjXF4/view?usp=sharing](https://drive.google.com/file/d/18pc9PptjfcjQfPyVCiaSq30RFs3ZjXF4/view?usp=sharing)", # معطل مؤقتاً
    # ... بقية الروابط معطلة مؤقتاً لتحسين سرعة النشر الأولي
]

# -------------------------------------------------
#  2. تحميل النماذج وإعداد قاعدة البيانات (بشكل محسّن)
# -------------------------------------------------

@st.cache_resource
def load_embedding_model():
    """
    تحميل نموذج التضمين وتخزينه في الذاكرة المؤقتة.
    """
    try:
        return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    except Exception as e:
        st.error(f"فشل فادح في تحميل نموذج الذكاء الاصطناعي: {e}")
        st.stop()

@st.cache_resource
def init_chroma_db():
    """
    إعداد ChromaDB في مجلد مؤقت آمن على الخادم.
    """
    try:
        # استخدام مجلد مؤقت لضمان التوافق مع أي بيئة تشغيل
        temp_dir = tempfile.gettempdir()
        db_path = os.path.join(temp_dir, "chroma_db_godfather")
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_or_create_collection(name="pigeon_genetics_knowledge")
        return collection
    except Exception as e:
        st.error(f"فشل فادح في تهيئة قاعدة البيانات: {e}")
        st.stop()

@st.cache_data(ttl=3600) # تخزين البيانات لمدة ساعة
def build_knowledge_base(_collection, _model):
    """
    بناء قاعدة المعرفة فقط إذا كانت فارغة.
    """
    if _collection.count() == 0:
        with st.status("⚙️ يتم بناء قاعدة المعرفة لأول مرة...", expanded=True) as status:
            all_chunks, all_metadata, all_ids = [], [], []
            doc_id_counter = 0

            for i, link in enumerate(BOOK_LINKS):
                status.update(label=f"جاري معالجة الكتاب {i+1}/{len(BOOK_LINKS)}...")
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                        file_id = link.split('/d/')[1].split('/')[0]
                        gdown.download(id=file_id, output=tmp.name, quiet=True)
                        
                        text = ""
                        with open(tmp.name, 'rb') as f:
                            reader = PyPDF2.PdfReader(f)
                            for page in reader.pages:
                                text += (page.extract_text() or "") + "\n"
                        
                        # تقسيم النص إلى أجزاء
                        chunks = text.split('\n\n')
                        for chunk in chunks:
                            if len(chunk.strip()) > 150:
                                all_chunks.append(chunk.strip())
                                all_metadata.append({'source': link})
                                all_ids.append(f"doc_{doc_id_counter}")
                                doc_id_counter += 1
                except Exception as e:
                    st.warning(f"حدث خطأ أثناء معالجة الكتاب {i+1}. سيتم تخطيه. الخطأ: {e}")
                finally:
                    if 'tmp' in locals() and os.path.exists(tmp.name):
                        os.remove(tmp.name)

            if all_chunks:
                status.update(label="جاري تحويل النصوص إلى متجهات...")
                embeddings = _model.encode(all_chunks).tolist()
                _collection.add(embeddings=embeddings, documents=all_chunks, metadatas=all_metadata, ids=all_ids)
            
            status.update(label="✅ اكتمل بناء قاعدة المعرفة بنجاح!", state="complete")
    return True

# -------------------------------------------------
#  3. واجهة المستخدم الرئيسية
# -------------------------------------------------
st.title("🕊️ العرّاب للجينات - الإصدار 3.1 (مستقر)")
st.write("ابحث في المراجع العلمية لوراثة الحمام.")

# تحميل المكونات الأساسية
model = load_embedding_model()
db_collection = init_chroma_db()
build_knowledge_base(db_collection, model)

# مربع البحث
query = st.text_input("اكتب سؤالك هنا:", placeholder="مثال: ما هو تأثير جين Spread؟")

if query:
    with st.spinner("جاري البحث..."):
        # تحويل السؤال إلى متجه للبحث
        query_embedding = model.encode([query]).tolist()
        results = db_collection.query(query_embeddings=query_embedding, n_results=3)

        documents = results.get('documents', [[]])[0]
        if documents:
            for i, doc in enumerate(documents):
                similarity = (1 - results['distances'][0][i]) * 100
                source = results['metadatas'][0][i]['source']
                
                if i == 0:
                    st.success(f"🔍 أفضل نتيجة (بنسبة تشابه ~{similarity:.0f}%):")
                    st.markdown(f"> {doc}")
                    st.caption(f"المصدر: {source}")
                else:
                    with st.expander(f"نتيجة إضافية (بنسبة تشابه ~{similarity:.0f}%)"):
                        st.info(doc)
                        st.caption(f"المصدر: {source}")
        else:
            st.warning("لم يتم العثور على نتائج مطابقة في قاعدة المعرفة الحالية.")
