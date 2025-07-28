# ==============================================================================
#  HOT-PATCH FOR SQLITE3 VERSION ON STREAMLIT CLOUD
# ==============================================================================
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ==============================================================================


# ==============================================================================
#  مشروع العرّاب للجينات - الإصدار 4.0 (مع الترجمة والواجهة المحسنة)
# ==============================================================================

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import gdown
import PyPDF2
import os
import tempfile
import requests # لإجراء طلبات API
import json
import shutil # لإضافة shutil.rmtree

# -------------------------------------------------
#  1. إعدادات الصفحة والمصادر
# -------------------------------------------------
st.set_page_config(
    page_title="العرّاب للجينات",
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
#  2. تحميل النماذج وإعداد قاعدة البيانات
# -------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

@st.cache_resource
def init_chroma_db(): # إزالة embedding_model كمعامل
    temp_dir = tempfile.gettempdir()
    db_path = os.path.join(temp_dir, "chroma_db_godfather")
    
    # حذف قاعدة البيانات الموجودة إذا كانت موجودة (للتأكد من إعادة البناء النظيف)
    if os.path.exists(db_path):
        try:
            shutil.rmtree(db_path)
            st.warning("تم حذف قاعدة بيانات ChromaDB القديمة لإعادة البناء.")
        except Exception as e:
            st.error(f"فشل حذف قاعدة بيانات ChromaDB القديمة: {e}")

    client = chromadb.PersistentClient(path=db_path)
    # لا تمرر embedding_function هنا
    return client.get_or_create_collection(name="pigeon_genetics_knowledge")

@st.cache_data(ttl=3600)
def build_knowledge_base(_collection, model): # إضافة model كمعامل
    if _collection.count() == 0:
        with st.status("⚙️ يتم بناء قاعدة المعرفة الكاملة لأول مرة...", expanded=True) as status:
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
                        chunks = text.split('\n\n')
                        for chunk in chunks:
                            if len(chunk.strip()) > 150:
                                all_chunks.append(chunk.strip())
                                all_metadata.append({'source': link})
                                all_ids.append(f"doc_{doc_id_counter}")
                                doc_id_counter += 1
                finally:
                    if 'tmp' in locals() and os.path.exists(tmp.name):
                        os.remove(tmp.name)
            if all_chunks:
                # تضمين النصوص يدوياً قبل الإضافة
                embeddings = model.encode(all_chunks).tolist()
                _collection.add(documents=all_chunks, metadatas=all_metadata, ids=all_ids, embeddings=embeddings)
            status.update(label="✅ اكتمل بناء قاعدة المعرفة!", state="complete")
    return True

# -------------------------------------------------
#  3. دالة الترجمة باستخدام Gemini API
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
        
        if result.get('candidates'):
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return "حدث خطأ أثناء الترجمة. قد يكون النص الأصلي هو الأفضل في هذه الحالة."
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return f"فشل الاتصال بخدمة الترجمة. النص الأصلي: {text_to_translate}"

# -------------------------------------------------
#  4. واجهة المستخدم الرئيسية
# -------------------------------------------------
st.title("🕊️ العرّاب للجينات - الإصدار 4.0")

# تحميل المكونات الأساسية
model = load_embedding_model()
db_collection = init_chroma_db() # لا تمرر model هنا
# تمرير model إلى build_knowledge_base
build_knowledge_base(db_collection, model)

tab1, tab2 = st.tabs(["🧠 المساعد الذكي", "🧬 الحاسبة الوراثية (قريباً)"])

with tab1:
    st.header("حوار مع خبير الوراثة")
    st.write("اطرح سؤالاً للحصول على إجابات مترجمة من المراجع العلمية.")
    
    query = st.text_input("اكتب سؤالك هنا:", placeholder="مثال: ما هو تأثير جين Spread؟", label_visibility="collapsed")

    if query:
        with st.spinner("جاري البحث في المراجع والترجمة..."):
            # 1. البحث في قاعدة المعرفة
            # تضمين الاستعلام يدوياً قبل البحث
            query_embedding = model.encode([query]).tolist()[0]
            results = db_collection.query(query_embeddings=[query_embedding], n_results=1)
            documents = results.get('documents', [[]])[0]

            if documents:
                # 2. أخذ أفضل نتيجة وترجمتها
                best_result_text = documents[0]
                source = results['metadatas'][0][0]['source']
                
                translated_text = translate_text_with_gemini(best_result_text)
                
                # 3. عرض النتيجة المترجمة
                st.success("**الإجابة (مترجمة من المصدر):**")
                st.markdown(f"<div dir='rtl' style='text-align: right;'>{translated_text}</div>", unsafe_allow_html=True)
                st.caption(f"المصدر الأصلي (باللغة الإنجليزية): {source}")

            else:
                st.warning("لم يتم العثور على نتائج مطابقة في قاعدة المعرفة الحالية.")

with tab2:
    st.header("الحاسبة الوراثية المتقدمة")
    st.info("سيتم تفعيل هذه الميزة في المرحلة القادمة من خارطة الطريق.")
    st.image("https://placehold.co/600x300/e2e8f0/4a5568?text=Genetic+Calculator+UI", caption="تصور لواجهة الحاسبة الوراثية")

