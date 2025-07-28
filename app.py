# ==============================================================================
#  مشروع العرّاب للجينات: الواجهة التفاعلية (بمحرك ChromaDB)
#  الإصدار 2.0: نسخة متوافقة مع Streamlit Cloud
# ==============================================================================

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import gdown
import PyPDF2
import os

# -------------------------------------------------
#  1. إعدادات الصفحة
# -------------------------------------------------
st.set_page_config(
    page_title="العرّاب للجينات",
    page_icon="🕊️",
    layout="wide",
)

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
#  2. تحميل النموذج
# -------------------------------------------------
@st.cache_resource
def load_embedding_model(model_name='paraphrase-multilingual-mpnet-base-v2'):
    return SentenceTransformer(model_name)

# -------------------------------------------------
#  3. إعداد قاعدة البيانات (ChromaDB)
# -------------------------------------------------
@st.cache_resource
def initialize_chroma_db(_model):
    client = chromadb.PersistentClient(path="chroma_db_store")
    collection = client.get_or_create_collection(name="pigeon_genetics_knowledge")

    if collection.count() == 0:
        with st.status("⏳ يتم بناء قاعدة المعرفة لأول مرة، يرجى الانتظار.", expanded=True) as status:
            st.write("الخطوة 1/3: تحميل المراجع العلمية (PDFs).")
            all_texts = []
            for link in BOOK_LINKS:
                try:
                    file_id = link.split('/d/')[1].split('/')[0]
                    output_filename = f"{file_id}.pdf"
                    gdown.download(id=file_id, output=output_filename, quiet=True)

                    text = ""
                    with open(output_filename, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        if reader.is_encrypted:
                            reader.decrypt("")
                        for page in reader.pages:
                            text += (page.extract_text() or "") + "\n"

                    all_texts.append({'source': link, 'content': text})
                    os.remove(output_filename)
                except Exception as e:
                    st.warning(f"فشل تحميل أو قراءة الكتاب: {link}. الخطأ: {e}")

            st.write("الخطوة 2/3: تقسيم النصوص.")
            all_chunks, all_metadata, all_ids = [], [], []
            doc_id_counter = 0
            for doc in all_texts:
                chunks = doc['content'].split('\n\n')
                for chunk in chunks:
                    if len(chunk.strip()) > 150:
                        all_chunks.append(chunk.strip())
                        all_metadata.append({'source': doc['source']})
                        all_ids.append(f"doc_{doc_id_counter}")
                        doc_id_counter += 1

            st.write("الخطوة 3/3: إضافة البيانات إلى قاعدة المعرفة.")
            if all_chunks:
                collection.add(
                    documents=all_chunks,
                    metadatas=all_metadata,
                    ids=all_ids
                )

            status.update(label="✅ اكتمل بناء قاعدة المعرفة بنجاح!", state="complete", expanded=False)

    return collection

# -------------------------------------------------
#  4. البحث في قاعدة المعرفة
# -------------------------------------------------
def search_knowledge_base(query, collection, n_results=5):
    return collection.query(query_texts=[query], n_results=n_results)

# -------------------------------------------------
#  5. واجهة المستخدم
# -------------------------------------------------
st.title("🕊️ العرّاب للجينات (ChromaDB)")
st.write("اكتب سؤالاً للبحث في مراجع وراثة الحمام.")

embedding_model = load_embedding_model()
knowledge_collection = initialize_chroma_db(embedding_model)

user_query = st.text_input("اسأل عن أي شيء في وراثة الحمام...", placeholder="مثال: ما هو جين الأوبال؟")

if user_query:
    with st.spinner("جاري البحث..."):
        search_results = search_knowledge_base(user_query, knowledge_collection)

    st.subheader("نتائج البحث:")
    documents = search_results.get('documents', [[]])[0]
    metadatas = search_results.get('metadatas', [[]])[0]
    distances = search_results.get('distances', [[]])[0]

    if not documents:
        st.warning("لم يتم العثور على إجابة.")
    else:
        for i, doc in enumerate(documents):
            source = metadatas[i].get('source', 'غير معروف')
            similarity = (1 - distances[i]) * 100
            if i == 0:
                st.success(f"**أفضل نتيجة (~{similarity:.0f}%):**")
                st.markdown(f"> {doc}")
                st.caption(f"المصدر: {source}")
            else:
                with st.expander(f"نتيجة إضافية (~{similarity:.0f}%)"):
                    st.info(doc)
                    st.caption(f"المصدر: {source}")
