# ==============================================================================
#  مشروع العرّاب للجينات: الواجهة التفاعلية
#  المرحلة 2: بناء الواجهة والتكامل
#  -- الإصدار 1.0: واجهة المحادثة الأساسية --
# ==============================================================================

# -------------------------------------------------
#  الخطوة 1: استيراد المكتبات اللازمة
# -------------------------------------------------
import streamlit as st
import sqlite3
import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# -------------------------------------------------
#  الخطوة 2: إعدادات الصفحة وتحميل النماذج
# -------------------------------------------------
# إعدادات أساسية لصفحة الويب
st.set_page_config(
    page_title="العرّاب للجينات",
    page_icon="🕊️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تحميل نموذج تحويل النصوص إلى متجهات (يجب أن يكون نفس النموذج المستخدم في البناء)
# سيتم تحميله مرة واحدة فقط وتخزينه في الذاكرة المؤقتة لتحسين الأداء
@st.cache_resource
def load_model(model_name='paraphrase-multilingual-mpnet-base-v2'):
    """
    تقوم بتحميل نموذج SentenceTransformer من الإنترنت.
    """
    print("جاري تحميل نموذج الذكاء الاصطناعي...")
    model = SentenceTransformer(model_name)
    print("اكتمل تحميل النموذج.")
    return model

# تحميل قاعدة البيانات المتجهية
@st.cache_data
def load_vector_store(file_path="vector_store.pkl"):
    """
    تقوم بتحميل قاعدة البيانات المتجهية من الملف.
    """
    print("جاري تحميل قاعدة البيانات المتجهية (ذاكرة الكتب)...")
    try:
        with open(file_path, "rb") as f:
            vector_store = pickle.load(f)
        print("اكتمل تحميل قاعدة البيانات المتجهية.")
        return vector_store
    except FileNotFoundError:
        st.error(f"خطأ: لم يتم العثور على ملف قاعدة البيانات المتجهية '{file_path}'. يرجى التأكد من تشغيل سكربت بناء قاعدة البيانات أولاً.")
        return None

# -------------------------------------------------
#  الخطوة 3: دوال البحث والاستعلام
# -------------------------------------------------
def search_knowledge_base(query, model, vector_store, top_k=3):
    """
    تبحث عن إجابة لسؤال المستخدم داخل قاعدة البيانات المتجهية.
    """
    if vector_store is None:
        return []

    print(f"جاري البحث عن إجابة لـ: '{query}'")
    # تحويل سؤال المستخدم إلى متجه
    query_embedding = model.encode([query], convert_to_tensor=True)
    query_embedding = query_embedding.cpu().detach().numpy().astype('float32')

    # البحث في فهرس FAISS عن أقرب المتجهات
    # D: مصفوفة المسافات (مدى القرب)، I: مصفوفة المؤشرات (أرقام الأجزاء)
    distances, indices = vector_store['index'].search(query_embedding, top_k)
    
    # تجميع النتائج
    results = []
    for i in range(top_k):
        chunk_index = indices[0][i]
        results.append({
            "text": vector_store['chunks'][chunk_index],
            "source": vector_store['metadata'][chunk_index],
            "score": 1 - distances[0][i] # تحويل المسافة إلى درجة تشابه
        })
    
    print("اكتمل البحث.")
    return results

# -------------------------------------------------
#  الخطوة 4: بناء واجهة المستخدم
# -------------------------------------------------
def main():
    # العنوان الرئيسي للتطبيق
    st.title("🕊️ العرّاب للجينات: وكيلك الذكي لوراثة الحمام")
    st.write("اطرح سؤالاً للحصول على إجابات من قاعدة المعرفة العلمية التي بنيناها.")

    # تحميل المكونات الأساسية
    model = load_model()
    vector_store = load_vector_store()

    # مربع إدخال السؤال
    user_query = st.text_input("اسأل عن أي شيء في وراثة الحمام...", placeholder="مثال: ما هو جين الأوبال السائد؟")

    if user_query:
        # البحث عن الإجابة عند إدخال المستخدم لسؤال
        with st.spinner("جاري البحث في المراجع العلمية..."):
            search_results = search_knowledge_base(user_query, model, vector_store)

        st.subheader("نتائج البحث:")
        
        if not search_results:
            st.warning("لم أتمكن من العثور على إجابة دقيقة في قاعدة المعرفة الحالية.")
        else:
            # عرض أفضل نتيجة بشكل مميز
            best_result = search_results[0]
            st.success(f"**أفضل نتيجة (بنسبة تشابه {best_result['score']:.2%}):**")
            st.markdown(f"> {best_result['text']}")
            st.caption(f"المصدر: {best_result['source']}")
            
            # عرض النتائج الأخرى (إذا وجدت)
            if len(search_results) > 1:
                with st.expander("عرض نتائج إضافية"):
                    for result in search_results[1:]:
                        st.info(result['text'])
                        st.caption(f"المصدر: {result['source']} (بنسبة تشابه {result['score']:.2%})")

# -------------------------------------------------
#  الخطوة 5: تشغيل التطبيق
# -------------------------------------------------
if __name__ == "__main__":
    main()
