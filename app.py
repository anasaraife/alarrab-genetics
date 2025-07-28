# ==============================================================================
#  العرّاب للجينات - الإصدار المحسن 5.0 (حل مشاكل ChromaDB)
# ==============================================================================

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gdown
import PyPDF2
import os
import tempfile
import hashlib
from datetime import datetime

# -------------------------------------------------
#  1. إعدادات الصفحة والمصادر
# -------------------------------------------------
st.set_page_config(
    page_title="العرّاب للجينات - الإصدار المحسن",
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

# قاموس الجينات الأساسية (بيانات محلية كاحتياطي)
GENETICS_DATABASE = {
    "genes": {
        "Blue/Black": {
            "symbol": "B+", "chromosome": "Z", "inheritance": "Sex-linked",
            "description": "الجين المسؤول عن اللون الأزرق/الأسود في الحمام",
            "phenotype": "لون أزرق رمادي أو أسود حسب وجود جينات أخرى"
        },
        "Ash-red": {
            "symbol": "BA", "chromosome": "Z", "inheritance": "Sex-linked",
            "description": "الجين المسؤول عن اللون الأحمر الرمادي",
            "phenotype": "لون أحمر رمادي مع تدرجات مختلفة"
        },
        "Brown": {
            "symbol": "b", "chromosome": "Z", "inheritance": "Sex-linked",
            "description": "الجين المسؤول عن اللون البني",
            "phenotype": "لون بني شوكولاتي"
        },
        "Checker": {
            "symbol": "C", "chromosome": "1", "inheritance": "Autosomal",
            "description": "نمط الشطرنج على الأجنحة",
            "phenotype": "نمط مربعات داكنة على الأجنحة"
        },
        "Red Bar": {
            "symbol": "T", "chromosome": "1", "inheritance": "Autosomal",
            "description": "الخطوط الحمراء على الأجنحة",
            "phenotype": "خطان أحمران عرضيان على كل جناح"
        },
        "Spread": {
            "symbol": "S", "chromosome": "8", "inheritance": "Autosomal",
            "description": "انتشار اللون على كامل الطائر",
            "phenotype": "لون موحد بدون أنماط أو خطوط"
        }
    }
}

# -------------------------------------------------
#  2. قاعدة البيانات البديلة (SQLite + TF-IDF)
# -------------------------------------------------

@st.cache_resource
def init_sqlite_db():
    """إنشاء قاعدة بيانات SQLite كبديل لـ ChromaDB"""
    db_path = os.path.join(tempfile.gettempdir(), "genetics_knowledge.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            source TEXT NOT NULL,
            content_hash TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cached_queries (
            query_hash TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    return conn

# -------------------------------------------------
#  3. بناء قاعدة المعرفة البديلة
# -------------------------------------------------

@st.cache_data(ttl=7200)
def build_knowledge_base_sqlite(_conn):
    """بناء قاعدة المعرفة باستخدام SQLite"""
    cursor = _conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM knowledge_base")
    count = cursor.fetchone()[0]
    
    if count == 0:
        with st.status("⚙️ يتم بناء قاعدة المعرفة...", expanded=True) as status:
            documents_added = 0
            
            # إضافة البيانات المحلية أولاً
            for gene_name, gene_info in GENETICS_DATABASE["genes"].items():
                content = f"Gene: {gene_name}\nSymbol: {gene_info['symbol']}\nChromosome: {gene_info['chromosome']}\nInheritance: {gene_info['inheritance']}\nDescription: {gene_info['description']}\nPhenotype: {gene_info['phenotype']}"
                content_hash = hashlib.md5(content.encode()).hexdigest()
                try:
                    cursor.execute("INSERT OR IGNORE INTO knowledge_base (content, source, content_hash) VALUES (?, ?, ?)", (content, "Local Database", content_hash))
                    documents_added += 1
                except sqlite3.IntegrityError:
                    pass
            
            # محاولة تحميل من الكتب
            for i, link in enumerate(BOOK_LINKS[:3]):
                status.update(label=f"جاري معالجة الكتاب {i+1}/3...")
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                        file_id = link.split('/d/')[1].split('/')[0]
                        gdown.download(id=file_id, output=tmp.name, quiet=True)
                        
                        with open(tmp.name, 'rb') as f:
                            reader = PyPDF2.PdfReader(f)
                            for page_num, page in enumerate(reader.pages[:10]):
                                text = page.extract_text() or ""
                                if len(text.strip()) > 100:
                                    content_hash = hashlib.md5(text.encode()).hexdigest()
                                    try:
                                        cursor.execute("INSERT OR IGNORE INTO knowledge_base (content, source, content_hash) VALUES (?, ?, ?)", (text.strip(), f"Book_{i+1}_Page_{page_num+1}", content_hash))
                                        documents_added += 1
                                    except sqlite3.IntegrityError:
                                        pass
                        os.remove(tmp.name)
                except Exception as e:
                    st.warning(f"تعذر تحميل الكتاب {i+1}: {str(e)}")
                    continue
            
            _conn.commit()
            status.update(label=f"✅ تم إضافة {documents_added} وثيقة لقاعدة المعرفة!", state="complete")
    
    return True

# -------------------------------------------------
#  4. البحث الذكي باستخدام TF-IDF
# -------------------------------------------------

def search_knowledge_base(query, conn, limit=3):
    """البحث في قاعدة المعرفة باستخدام TF-IDF"""
    cursor = conn.cursor()
    cursor.execute("SELECT id, content, source FROM knowledge_base")
    results = cursor.fetchall()
    
    if not results:
        return []
    
    documents = [row[1] for row in results]
    sources = [row[2] for row in results]
    
    try:
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        tfidf_matrix = vectorizer.fit_transform(documents)
        query_vector = vectorizer.transform([query])
        
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        top_indices = similarities.argsort()[-limit:][::-1]
        
        search_results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                search_results.append({
                    'content': documents[idx],
                    'source': sources[idx],
                    'score': similarities[idx]
                })
        
        return search_results
    
    except Exception as e:
        st.error(f"خطأ في البحث: {str(e)}")
        return []

# -------------------------------------------------
#  5. الترجمة المبسطة
# -------------------------------------------------

def simple_translate_genetics_terms(text):
    """ترجمة مبسطة للمصطلحات الوراثية الأساسية"""
    translation_dict = {
        "gene": "جين", "allele": "أليل", "chromosome": "كروموسوم", "dominant": "سائد",
        "recessive": "متنحي", "phenotype": "نمط ظاهري", "genotype": "نمط وراثي",
        "inheritance": "وراثة", "mutation": "طفرة", "breeding": "تزاوج", "pigeon": "حمام",
        "color": "لون", "pattern": "نمط", "blue": "أزرق", "red": "أحمر", "black": "أسود",
        "brown": "بني", "white": "أبيض", "spread": "انتشار", "checker": "شطرنج", "bar": "خط"
    }
    
    translated_text = text
    for english, arabic in translation_dict.items():
        translated_text = translated_text.replace(english.title(), arabic)
        translated_text = translated_text.replace(english.lower(), arabic)
    
    return translated_text

# -------------------------------------------------
#  6. واجهة المستخدم المحسنة
# -------------------------------------------------

st.title("🕊️ العرّاب للجينات - الإصدار المحسن 5.0")
st.markdown("*نظام ذكي لاستكشاف وراثة الحمام مع حلول محسنة للاستقرار*")

try:
    db_conn = init_sqlite_db()
    build_knowledge_base_sqlite(db_conn)
    
    with st.sidebar:
        st.header("📊 إحصائيات النظام")
        cursor = db_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM knowledge_base")
        doc_count = cursor.fetchone()[0]
        st.metric("عدد الوثائق", doc_count)
        
        st.header("🧬 الجينات المتاحة")
        for gene_name in GENETICS_DATABASE["genes"].keys():
            st.write(f"• {gene_name}")

    tab1, tab2, tab3 = st.tabs(["🤖 المساعد الذكي", "🧬 موسوعة الجينات", "📊 لوحة التحكم"])

    with tab1:
        st.header("المساعد الذكي للوراثة")
        
        example_buttons = st.columns(3)
        if 'example_query' not in st.session_state:
            st.session_state.example_query = ""

        if example_buttons[0].button("ما هو جين Spread؟"):
            st.session_state.example_query = "What is Spread gene?"
        if example_buttons[1].button("كيف يورث اللون الأزرق؟"):
            st.session_state.example_query = "How is blue color inherited?"
        if example_buttons[2].button("ما هو نمط الشطرنج؟"):
            st.session_state.example_query = "What is checker pattern?"
        
        query = st.text_input("اطرح سؤالك:", value=st.session_state.example_query, placeholder="مثال: ما تأثير جين Ash-red على لون الحمام؟")
        
        if query:
            with st.spinner("جاري البحث..."):
                results = search_knowledge_base(query, db_conn)
                
                if results:
                    st.success("**النتائج الموجودة:**")
                    for i, result in enumerate(results[:2]):
                        with st.expander(f"النتيجة {i+1} (درجة التطابق: {result['score']:.2f})"):
                            translated_content = simple_translate_genetics_terms(result['content'])
                            st.write(translated_content)
                            st.caption(f"المصدر: {result['source']}")
                else:
                    st.warning("لم يتم العثور على نتائج مطابقة.")
                    st.info("💡 جرب البحث عن مصطلحات مثل: Blue, Red, Spread, Checker, أو أسماء الجينات الإنجليزية.")

    with tab2:
        st.header("موسوعة الجينات")
        for gene_name, gene_info in GENETICS_DATABASE["genes"].items():
            with st.expander(f"🧬 {gene_name} ({gene_info['symbol']})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**المعلومات الأساسية:**")
                    st.write(f"• الرمز: `{gene_info['symbol']}`")
                    st.write(f"• الكروموسوم: {gene_info['chromosome']}")
                    st.write(f"• نوع الوراثة: {gene_info['inheritance']}")
                with col2:
                    st.write("**الوصف:**")
                    st.write(gene_info['description'])
                    st.write("**النمط الظاهري:**")
                    st.write(gene_info['phenotype'])

    with tab3:
        st.header("لوحة تحكم النظام")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("حالة قاعدة البيانات", "✅ متصلة")
        with col2:
            cursor = db_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM knowledge_base")
            total_docs = cursor.fetchone()[0]
            st.metric("إجمالي المراجع", total_docs)

except Exception as e:
    st.error(f"خطأ في تشغيل النظام: {str(e)}")
    st.info("يتم تشغيل النظام في الوضع الآمن مع البيانات المحلية فقط.")
    
    st.header("🧬 البيانات المحلية")
    for gene_name, gene_info in GENETICS_DATABASE["genes"].items():
        with st.expander(f"{gene_name} ({gene_info['symbol']})"):
            st.write(f"**الوصف:** {gene_info['description']}")
            st.write(f"**النمط الظاهري:** {gene_info['phenotype']}")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'><p>🕊️ العرّاب للجينات - الإصدار 5.0 المحسن</p></div>", unsafe_allow_html=True)
