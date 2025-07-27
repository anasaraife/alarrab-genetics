# ===================================================================
# 🚀 العرّاب للجينات V7.0 - الذاكرة المتطورة
# ميزة حصرية: لوحة تحكم للأدمن لتغذية الوكيل بمعرفة جديدة
# يتم حفظها بشكل دائم في Google Drive.
# ===================================================================

import streamlit as st
import os
import re
import pickle
import json
from datetime import datetime
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
import numpy as np

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

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False

# --- 1. إعدادات الصفحة ---
st.set_page_config(
    layout="wide",
    page_title="العرّاب للجينات V7.0",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# --- 2. CSS مخصص للواجهة ---
st.markdown("""
<style>
    .main-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem; }
    .admin-panel { background: #fff3cd; border-left: 5px solid #ffc107; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# --- 3. إدارة الجلسة والذاكرة الديناميكية ---
def initialize_session_state():
    """تهيئة حالة الجلسة."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False
    if "resources" not in st.session_state:
        st.session_state.resources = load_initial_resources()
    if "model" not in st.session_state:
        st.session_state.model = initialize_enhanced_gemini()

# --- 4. تحميل الموارد المتقدم ---
@st.cache_resource(show_spinner="جاري تحميل الموارد الأساسية...")
def load_initial_resources():
    """تحميل الموارد الأولية مرة واحدة عند بدء التطبيق."""
    resources = {"status": "loading"}
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
            st.warning("ملف قاعدة المعرفة (vector_db.pkl) غير موجود.")
            resources["status"] = "no_db"
    else:
        resources["status"] = "vector_search_unavailable"
    return resources

@st.cache_resource(show_spinner="جاري تهيئة الذكاء الاصطناعي...")
def initialize_enhanced_gemini():
    """تهيئة نموذج Gemini."""
    if not GEMINI_AVAILABLE: return None
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key: return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash', generation_config={"temperature": 0.1, "max_output_tokens": 4096})
    except Exception as e:
        st.error(f"فشل تهيئة Gemini: {e}")
        return None

# --- 5. أدوات الأدمن للتعلم الدائم ---
def get_gdrive_service():
    """إنشاء خدمة Google Drive للتواصل مع API."""
    if not GDRIVE_AVAILABLE:
        st.error("مكتبة gspread غير متاحة. لا يمكن الاتصال بـ Google Drive.")
        return None
    try:
        # استخدام بيانات الاعتماد من أسرار Streamlit
        creds_json = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(
            creds_json,
            scopes=["https://www.googleapis.com/auth/drive"]
        )
        from googleapiclient.discovery import build
        service = build('drive', 'v3', credentials=creds)
        return service
    except Exception as e:
        st.error(f"❌ فشل الاتصال بـ Google Drive: {e}")
        st.info("تأكد من إعداد بيانات اعتماد حساب الخدمة (gcp_service_account) بشكل صحيح في أسرار التطبيق.")
        return None

def save_knowledge_to_drive(content: str, source_url: str):
    """حفظ المعرفة الجديدة كملف نصي في مجلد Google Drive."""
    service = get_gdrive_service()
    if not service:
        return

    folder_id = st.secrets.get("GDRIVE_FOLDER_ID")
    if not folder_id:
        st.error("معرف مجلد Google Drive (GDRIVE_FOLDER_ID) غير موجود في الأسرار.")
        return

    with st.spinner("💾 جاري حفظ المعرفة الجديدة بشكل دائم..."):
        try:
            from googleapiclient.http import MediaIoBaseUpload
            import io

            file_metadata = {
                'name': f'knowledge_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
                'parents': [folder_id]
            }
            
            full_content = f"Source URL: {source_url}\n\n{content}"
            fh = io.BytesIO(full_content.encode('utf-8'))
            
            media = MediaIoBaseUpload(fh, mimetype='text/plain')
            
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            st.success(f"✅ تم حفظ المعرفة بنجاح في Google Drive! (معرف الملف: {file.get('id')})")
        except Exception as e:
            st.error(f"❌ فشل حفظ المعرفة في Google Drive: {e}")


def scrape_and_process_url(url: str) -> List[str]:
    """تصفح رابط واستخلاص وتنظيف وتقسيم النص."""
    try:
        st.info(f"🌐 جاري تصفح الرابط: {url}...")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.extract()
        
        text = soup.get_text(separator='\n', strip=True)
        cleaned_text = re.sub(r'\s+', ' ', text)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = text_splitter.split_text(cleaned_text)
        st.success(f"✅ تم استخلاص {len(chunks)} مقطع معرفي من الرابط.")
        return chunks, cleaned_text
    except Exception as e:
        st.error(f"❌ فشل استخلاص المعرفة من الرابط: {e}")
        return [], ""

def add_to_knowledge_base(new_chunks: List[str]):
    """إضافة المقاطع المعرفية الجديدة إلى الذاكرة الحية للجلسة الحالية."""
    if not VECTOR_SEARCH_AVAILABLE or 'embedder' not in st.session_state.resources:
        st.error("نظام البحث المتجه غير متاح لإضافة المعرفة.")
        return

    embedder = st.session_state.resources["embedder"]
    vector_db = st.session_state.resources["vector_db"]

    if not vector_db:
        st.error("قاعدة المعرفة الأولية غير موجودة.")
        return

    with st.spinner("🧠 الوكيل يتعلم... (جاري تحديث الذاكرة الحية)"):
        new_embeddings = embedder.encode(new_chunks)
        vector_db["index"].add(np.array(new_embeddings, dtype=np.float32))
        vector_db["chunks"].extend(new_chunks)
        st.session_state.resources["vector_db"] = vector_db
        st.success(f"💡 تم تحديث الذاكرة الحية بـ {len(new_chunks)} معلومة جديدة!")

# --- 6. الوكيل الذكي (يبقى كما هو) ---
class IntelligentGeneticAgent:
    def __init__(self, resources: dict):
        self.resources = resources

    def search_deep_memory(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.resources.get("vector_db") or not self.resources.get("embedder"): return []
        try:
            index = self.resources["vector_db"]["index"]
            chunks = self.resources["vector_db"]["chunks"]
            query_embedding = self.resources["embedder"].encode([query])
            distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
            return [{"content": chunks[idx], "score": 1 / (1 + dist)} for dist, idx in zip(distances[0], indices[0]) if idx < len(chunks)]
        except: return []

    def generate_smart_response(self, query: str) -> Dict:
        if not self.resources.get("model"):
            return {"answer": "❌ نظام الذكاء الاصطناعي غير متاح."}
        
        search_results = self.search_deep_memory(query)
        context_text = "\n\n---\n\n".join([r['content'] for r in search_results[:3]])
        
        prompt = f"""
أنت "العرّاب V7.0 - الخبير الموثوق". معرفتك مبنية على مكتبة رقمية متخصصة.
**السياق المرجعي:**
---
{context_text}
---
**سؤال المستخدم:** {query}
**تعليمات:** أجب بالاعتماد على السياق. إذا كانت المعلومة غير موجودة، قل ذلك بوضوح.
**التحليل:**
"""
        try:
            ai_response = self.resources["model"].generate_content(prompt)
            return {"answer": ai_response.text, "sources": search_results}
        except Exception as e:
            return {"answer": f"❌ خطأ في التحليل: {str(e)}", "sources": search_results}

# --- 7. واجهة المستخدم الشاملة ---
def main():
    initialize_session_state()
    
    # --- الشريط الجانبي ---
    with st.sidebar:
        st.markdown("## 🔑 لوحة تحكم الأدمن")
        admin_password = st.secrets.get("ADMIN_PASSWORD", "admin123")
        password = st.text_input("أدخل كلمة مرور الأدمن:", type="password")
        
        if password == admin_password:
            st.session_state.admin_authenticated = True
            st.success("✅ تم تسجيل الدخول بنجاح!")
        elif password:
            st.error("كلمة مرور خاطئة.")

        if st.session_state.admin_authenticated:
            st.markdown("---")
            st.markdown("### 🧠 تغذية الوكيل المعرفية")
            st.info("أضف رابطًا لمقال أو بحث لتعليم الوكيل وحفظه بشكل دائم.")
            
            url_to_learn = st.text_input("أدخل الرابط هنا:")
            if st.button("علّم الوكيل الآن", type="primary"):
                if url_to_learn:
                    new_chunks, full_content = scrape_and_process_url(url_to_learn)
                    if new_chunks:
                        # تحديث الذاكرة الحية
                        add_to_knowledge_base(new_chunks)
                        # حفظ المعرفة بشكل دائم في Drive
                        save_knowledge_to_drive(full_content, url_to_learn)
                else:
                    st.warning("الرجاء إدخال رابط.")
            
            st.markdown("---")
            st.markdown("### 📊 حالة الذاكرة الحية")
            if st.session_state.resources.get("vector_db"):
                current_chunks = len(st.session_state.resources["vector_db"]["chunks"])
                st.metric("عدد المقاطع المعرفية الحالية:", current_chunks)

    # --- الواجهة الرئيسية ---
    st.markdown('<div class="main-header"><h1>🚀 العرّاب للجينات V7.0</h1><p><strong>العقل المتطور - مع خاصية التعلم الدائم</strong></p></div>', unsafe_allow_html=True)
    
    agent = st.session_state.agent if "agent" in st.session_state else IntelligentGeneticAgent(st.session_state.resources)
    st.session_state.agent = agent

    chat_container = st.container(height=500)
    for message in st.session_state.messages:
        with chat_container.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("اسألني أي شيء..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        chat_container.chat_message("user").markdown(prompt)
        
        with chat_container.chat_message("assistant"):
            with st.spinner("🧠 العرّاب يفكر..."):
                response_data = agent.generate_smart_response(prompt)
                st.markdown(response_data["answer"])
                st.session_state.messages.append({"role": "assistant", "content": response_data["answer"]})

if __name__ == "__main__":
    main()
