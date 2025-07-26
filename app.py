# ===================================================================
# 🚀 العرّاب للجينات V4.0 - النسخة المحسنة والمتطورة
# مميزات جديدة: رسوم بيانية، تصدير، ذاكرة محادثة، واجهة محسنة
# ===================================================================

import streamlit as st
from itertools import product
import collections
import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime
from typing import List, Dict
import plotly.express as px
import hashlib
import time

# --- التحقق من توفر المكتبات ---
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False

# --- 1. إعدادات الصفحة ---
st.set_page_config(
    layout="wide",
    page_title="العرّاب للجينات V4.0",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# --- 2. CSS مخصص للواجهة ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. إدارة الجلسة ---
def initialize_session_state():
    """تهيئة حالة الجلسة."""
    defaults = {
        "messages": [],
        "search_history": [],
        "conversation_id": hashlib.md5(str(time.time()).encode()).hexdigest()[:8],
        "session_stats": {
            "queries_count": 0,
            "successful_searches": 0,
            "charts_generated": 0
        }
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- 4. تحميل الموارد ---
@st.cache_resource(show_spinner="جاري تحميل الموارد المعرفية...")
def load_enhanced_resources():
    """تحميل الموارد المعرفية والنموذج."""
    resources = {
        "vector_db": None, "embedder": None, "model": None,
        "status": "loading", "backup_data": None
    }
    
    if VECTOR_SEARCH_AVAILABLE:
        vector_db_path = "vector_db.pkl"
        if os.path.exists(vector_db_path):
            try:
                with open(vector_db_path, "rb") as f:
                    resources["vector_db"] = pickle.load(f)
                resources["embedder"] = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                resources["status"] = "ready"
            except Exception as e:
                st.error(f"خطأ في تحميل قاعدة المتجهات: {e}")
                resources["status"] = "failed"
        else:
            resources["status"] = "no_db_file"
    else:
        resources["status"] = "vector_search_unavailable"

    if GEMINI_AVAILABLE and "GEMINI_API_KEY" in st.secrets:
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            resources["model"] = genai.GenerativeModel('gemini-1.5-flash',
                generation_config={"temperature": 0.1, "max_output_tokens": 4096})
        except Exception as e:
            st.error(f"فشل تهيئة Gemini: {e}")
    
    if not resources["vector_db"]:
        resources["backup_data"] = create_sample_genetics_data()
        
    return resources

def create_sample_genetics_data():
    """إنشاء بيانات تجريبية."""
    return { "genetics_info": { "brown_gene": { "name": "الجين البني (Brown)", "symbol": "b", "description": "يتحكم في إنتاج الصبغة البنية.", "inheritance": "متنحي" }}}

# --- 5. وظائف البحث والوكيل الذكي ---
def search_knowledge(query: str, resources: dict, top_k: int = 5) -> List[Dict]:
    """البحث في قاعدة المعرفة."""
    if resources["vector_db"] and resources["embedder"]:
        index = resources["vector_db"]["index"]
        chunks = resources["vector_db"]["chunks"]
        metadata = resources["vector_db"].get("metadata", [])
        query_embedding = resources["embedder"].encode([query])
        distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
        return [{"content": chunks[idx], "metadata": metadata[idx] if idx < len(metadata) else {}, "score": float(1 - dist)} for dist, idx in zip(distances[0], indices[0]) if idx < len(chunks)]
    return []

def research_agent(query: str, resources: dict):
    """الوكيل البحثي المحلل."""
    if not resources.get("model"):
        return "❌ النظام غير مهيأ (API KEY مفقود أو غير صالح)."

    q_lower = query.lower().strip()
    if any(word in q_lower for word in ["سلام", "مرحبا", "اهلا", "هاي", "شكرا"]):
        return "🤗 وعليكم السلام! أنا العرّاب V4.0، كيف يمكنني مساعدتك اليوم؟"

    with st.spinner("🔬 جارٍ البحث والتحليل..."):
        search_results = search_knowledge(query, resources)
        if not search_results:
            return "🤔 لم أجد معلومات مباشرة. جرب إعادة صياغة السؤال."

        context_parts = []
        for i, result in enumerate(search_results):
            source = result['metadata'].get('source', 'غير معروف')
            page = result['metadata'].get('page', 'N/A')
            context_parts.append(f"[مصدر {i+1} من '{source}' صفحة {page}]:\n{result['content']}")
        context_text = "\n\n---\n\n".join(context_parts)

        prompt = f"""
        أنت "العرّاب الذكي V4.0"، خبير عالمي في علم وراثة الحمام.
        مهمتك هي تحليل المعلومات التالية وصياغة إجابة شاملة على سؤال المستخدم.

        **المعلومات المرجعية:**
        ---
        {context_text}
        ---
        **سؤال المستخدم:** {query}
        **تعليمات:** أجب بأسلوب علمي ومنظم، واستنتج الإجابة من خلال الربط بين المعلومات.
        **الإجابة التحليلية:**
        """
        try:
            response = resources["model"].generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ حدث خطأ أثناء صياغة الإجابة: {str(e)}"

# --- 6. واجهة المستخدم الرئيسية ---
def main():
    """الواجهة الرئيسية للتطبيق."""
    initialize_session_state()
    resources = load_enhanced_resources()

    st.markdown('<div class="main-header"><h1>🚀 العرّاب للجينات V4.0</h1><p>النسخة المحسنة - تحليل متقدم لوراثة وألوان الحمام بالذكاء الاصطناعي</p></div>', unsafe_allow_html=True)

    # الشريط الجانبي
    with st.sidebar:
        st.header("🔧 حالة النظام")
        if resources["status"] == "ready":
            st.success("✅ قاعدة المعرفة جاهزة")
            st.metric("📚 عدد المقاطع", len(resources["vector_db"]['chunks']))
        else:
            st.error("❌ قاعدة المعرفة غير متوفرة")
        
        if resources.get("model"):
            st.success("✅ Gemini متصل")
        else:
            st.error("❌ Gemini غير متاح")
        
        st.divider()
        st.header("📥 تصدير المحادثة")
        json_data = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)
        st.download_button(label="⬇️ تحميل JSON", data=json_data, file_name="conversation.json", mime="application/json")

    # واجهة المحادثة الرئيسية
    st.header("💬 المحادثة الذكية")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("مثال: ما هي الجينات المسؤولة عن اللون البني؟"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = research_agent(prompt, resources)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# --- 7. تشغيل التطبيق ---
if __name__ == "__main__":
    main()
