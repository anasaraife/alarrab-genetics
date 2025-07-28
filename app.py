# ==============================================================================
#  مشروع العرّاب للجينات - إعادة بناء من الصفر
# ==============================================================================

import streamlit as st
import gdown
import PyPDF2
import os
import tempfile
from sentence_transformers import SentenceTransformer
import faiss # استيراد FAISS
import numpy as np # لاستخدام NumPy مع FAISS
import requests # لإجراء طلبات API
import json

# -------------------------------------------------
#  1. إعدادات الصفحة والمصادر
# -------------------------------------------------
st.set_page_config(
    page_title="العرّاب للجينات",
    page_icon="🕊️",
    layout="wide",
)

# قائمة روابط الكتب (Google Drive PDF Links)
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
#  2. تحميل نموذج التضمين (Embedding Model)
# -------------------------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(\'paraphrase-multilingual-mpnet-base-v2\')

# -------------------------------------------------
#  3. دالة تحميل ومعالجة الكتب (PDF) وتوليد التضمينات
# -------------------------------------------------
@st.cache_data(ttl=3600)
def load_process_and_embed_books(model):
    all_chunks = []
    all_metadata = []
    all_ids = []
    doc_id_counter = 0

    with st.status("⚙️ جاري تحميل ومعالجة الكتب وتوليد التضمينات...", expanded=True) as status:
        for i, link in enumerate(BOOK_LINKS):
            status.update(label=f"جاري معالجة الكتاب {i+1}/{len(BOOK_LINKS)}...")
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=\".pdf\") as tmp:
                    file_id = link.split(\"/d/\")[1].split(\"/\")[0]
                    gdown.download(id=file_id, output=tmp.name, quiet=True)
                    text = ""
                    with open(tmp.name, \"rb\") as f:
                        reader = PyPDF2.PdfReader(f)
                        for page in reader.pages:
                            text += (page.extract_text() or "") + "\n"
                    
                    # تقسيم النص إلى أجزاء (chunks)
                    chunks = text.split(\'\\n\\n\') # تقسيم بناءً على سطرين فارغين
                    for chunk in chunks:
                        if len(chunk.strip()) > 150: # تصفية الأجزاء الصغيرة جداً
                            all_chunks.append(chunk.strip())
                            all_metadata.append({\'source\': link, \'book_index\': i+1})
                            all_ids.append(f"doc_{doc_id_counter}")
                            doc_id_counter += 1
            except Exception as e:
                st.error(f"فشل معالجة الكتاب {link}: {e}")
            finally:
                if \'tmp\' in locals() and os.path.exists(tmp.name):
                    os.remove(tmp.name)
        
        # توليد التضمينات لجميع الأجزاء
        if all_chunks:
            status.update(label="⚙️ جاري توليد التضمينات للنصوص...")
            all_embeddings = model.encode(all_chunks).tolist()
            status.update(label="✅ اكتمل تحميل ومعالجة الكتب وتوليد التضمينات!", state="complete")
            return all_chunks, all_metadata, all_ids, all_embeddings
        else:
            status.update(label="⚠️ لم يتم العثور على نصوص لمعالجتها.", state="complete")
            return [], [], [], []

# -------------------------------------------------
#  4. بناء قاعدة البيانات المتجهية (FAISS)
# -------------------------------------------------
@st.cache_resource
def build_faiss_index(embeddings):
    if not embeddings:
        return None
    # تحويل قائمة التضمينات إلى مصفوفة NumPy
    embeddings_array = np.array(embeddings).astype(\'float32\')
    # الحصول على أبعاد التضمين
    dimension = embeddings_array.shape[1]
    # بناء فهرس FAISS (هنا نستخدم IndexFlatL2 للبحث الدقيق)
    index = faiss.IndexFlatL2(dimension)
    # إضافة التضمينات إلى الفهرس
    index.add(embeddings_array)
    return index

# -------------------------------------------------
#  5. دالة الترجمة باستخدام Gemini API
# -------------------------------------------------
@st.cache_data
def translate_text_with_gemini(text_to_translate):
    """
    تستخدم Gemini API لترجمة النص إلى العربية.
    """
    API_KEY = "" # لا تحتاج إلى مفتاح للنماذج الأساسية
    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
    
    prompt = f"Translate the following technical text about pigeon genetics into clear and accurate Arabic. Keep the scientific terms if there is no common Arabic equivalent. Text to translate: \"{text_to_translate}\""
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    try:
        response = requests.post(API_URL, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        result = response.json()
        
        if result.get(\'candidates\'):
            return result[\'candidates\'][0][\'content\'][\'parts\'][0][\'text\']
        else:
            return "حدث خطأ أثناء الترجمة. قد يكون النص الأصلي هو الأفضل في هذه الحالة."
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return f"فشل الاتصال بخدمة الترجمة. النص الأصلي: {text_to_translate}"

# -------------------------------------------------
#  6. واجهة المستخدم الرئيسية
# -------------------------------------------------
st.title("🕊️ العرّاب للجينات - الإصدار 4.0")

# تحميل المكونات الأساسية
model = load_embedding_model()
chunks, metadata, ids, embeddings = load_process_and_embed_books(model)
faiss_index = build_faiss_index(embeddings)

tab1, tab2 = st.tabs(["🧠 المساعد الذكي", "🧬 الحاسبة الوراثية (قريباً)"])

with tab1:
    st.header("حوار مع خبير الوراثة")
    st.write("اطرح سؤالاً للحصول على إجابات مترجمة من المراجع العلمية.")
    
    query = st.text_input("اكتب سؤالك هنا:", placeholder="مثال: ما هو تأثير جين Spread؟", label_visibility="collapsed")

    if query:
        if faiss_index:
            with st.spinner("جاري البحث في المراجع والترجمة..."):
                # 1. البحث في قاعدة المعرفة
                query_embedding = model.encode([query]).astype(\'float32\')
                D, I = faiss_index.search(query_embedding, k=1) # k=1 للحصول على أقرب نتيجة
                
                if I[0][0] != -1: # التأكد من العثور على نتيجة
                    retrieved_chunk_index = I[0][0]
                    retrieved_chunk = chunks[retrieved_chunk_index]
                    retrieved_metadata = metadata[retrieved_chunk_index]
                    
                    translated_text = translate_text_with_gemini(retrieved_chunk)
                    
                    # 3. عرض النتيجة المترجمة
                    st.success("**الإجابة (مترجمة من المصدر):**")
                    st.markdown(f"<div dir=\'rtl\' style=\'text-align: right;\'>{translated_text}</div>", unsafe_allow_html=True)
                    st.caption(f"المصدر الأصلي: {retrieved_metadata[\'source\']}")

                else:
                    st.warning("لم يتم العثور على نتائج مطابقة في قاعدة المعرفة الحالية.")
        else:
            st.warning("لم يتم بناء فهرس FAISS. تأكد من وجود نصوص لمعالجتها.")

with tab2:
    st.header("الحاسبة الوراثية المتقدمة")
    st.info("سيتم تفعيل هذه الميزة في المرحلة القادمة من خارطة الطريق.")
    st.image("https://placehold.co/600x300/e2e8f0/4a5568?text=Genetic+Calculator+UI", caption="تصور لواجهة الحاسبة الوراثية")



