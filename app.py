# ===================================================================
# 🚀 العرّاب للجينات V2.0 - وكيل بحثي ذكي محسّن
# بميزات متقدمة: تحليل متعدد المستويات، ذاكرة محادثة، تصدير النتائج
# ===================================================================

import streamlit as st
from itertools import product
import collections
import pandas as pd
import google.generativeai as genai
import faiss
import numpy as np
import pickle
import os
import json
from datetime import datetime
import plotly.express as px
from typing import List, Dict

# --- 1. إعدادات الصفحة المحسّنة ---
st.set_page_config(
    layout="wide", 
    page_title="العرّاب للجينات V2.0",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# --- 2. فئات الأسئلة الذكية ---
QUESTION_CATEGORIES = {
    "basic": ["ما هو", "ما هي", "تعريف", "معنى"],
    "genetic_inheritance": ["وراثة", "جين", "كروموسوم", "دي ان ايه", "DNA"],
    "breeding": ["تربية", "تزاوج", "انتاج", "تحسين"],
    "colors": ["لون", "ألوان", "تلوين", "صبغة"],
    "diseases": ["مرض", "أمراض", "علاج", "صحة"],
    "analysis": ["حلل", "اشرح", "فسر", "قارن", "اربط"]
}

# --- 3. تحميل الموارد مع إدارة أفضل للأخطاء ---
@st.cache_resource
def load_resources():
    """تحميل قاعدة المتجهات ونموذج التضمين مع معالجة أفضل للأخطاء."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        st.error("❌ مكتبة sentence-transformers غير متوفرة")
        return None, None, None

    vector_db_path = "vector_db.pkl"
    metadata_path = "vector_metadata.json"
    embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # تحميل قاعدة المتجهات
    vector_db = None
    metadata = {}
    
    if os.path.exists(vector_db_path):
        try:
            with open(vector_db_path, "rb") as f:
                vector_db = pickle.load(f)
        except Exception as e:
            st.error(f"خطأ في تحميل قاعدة المتجهات: {e}")
    
    # تحميل البيانات الوصفية
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            st.warning(f"لم يتم تحميل البيانات الوصفية: {e}")
    
    # تحميل نموذج التضمين
    try:
        embedder = SentenceTransformer(embedding_model_name)
    except Exception as e:
        st.error(f"فشل تحميل نموذج التضمين: {e}")
        return None, None, None
    
    return vector_db, embedder, metadata

# --- 4. إعداد نموذج Gemini المحسّن ---
@st.cache_resource
def initialize_gemini():
    """تهيئة نموذج Gemini مع إعدادات محسّنة."""
    if "GEMINI_API_KEY" not in st.secrets:
        return None
    
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config={
                "temperature": 0.1,  # أقل عشوائية للدقة العلمية
                "max_output_tokens": 3000,
                "top_p": 0.8,
                "top_k": 40
            }
        )
        return model
    except Exception as e:
        st.error(f"فشل تهيئة Gemini: {e}")
        return None

# تحميل الموارد
vector_db, embedder, metadata = load_resources()
model = initialize_gemini()

# --- 5. وظائف البحث المتقدمة ---
def classify_question(query: str) -> str:
    """تصنيف السؤال حسب النوع لتحسين استراتيجية البحث."""
    query_lower = query.lower()
    
    for category, keywords in QUESTION_CATEGORIES.items():
        if any(keyword in query_lower for keyword in keywords):
            return category
    
    return "general"

def search_knowledge_advanced(query: str, category: str = "general", top_k: int = 5) -> List[Dict]:
    """بحث متقدم مع تصنيف ذكي للنتائج."""
    if not vector_db or not embedder:
        return []
    
    index = vector_db["index"]
    chunks = vector_db["chunks"]
    
    # تحسين الاستعلام حسب الفئة
    enhanced_query = enhance_query_by_category(query, category)
    
    # تحويل السؤال إلى متجه
    query_embedding = embedder.encode([enhanced_query])
    
    # البحث في FAISS مع المزيد من النتائج للفلترة
    search_k = min(top_k * 2, len(chunks))
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), search_k)
    
    # إنشاء نتائج مع درجات الصلة
    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(chunks):
            results.append({
                "content": chunks[idx],
                "relevance_score": 1 / (1 + dist),  # تحويل المسافة إلى درجة صلة
                "rank": i + 1,
                "source_info": metadata.get(str(idx), {})
            })
    
    # فلترة وترتيب النتائج
    results = filter_and_rank_results(results, query, category)
    
    return results[:top_k]

def enhance_query_by_category(query: str, category: str) -> str:
    """تحسين الاستعلام بناءً على فئة السؤال."""
    enhancements = {
        "genetic_inheritance": f"{query} وراثة جينات",
        "breeding": f"{query} تربية تزاوج",
        "colors": f"{query} ألوان وراثة",
        "diseases": f"{query} أمراض علاج",
        "analysis": f"{query} تحليل شرح"
    }
    
    return enhancements.get(category, query)

def filter_and_rank_results(results: List[Dict], query: str, category: str) -> List[Dict]:
    """فلترة وترتيب النتائج بناءً على معايير متقدمة."""
    # حساب درجة إضافية بناءً على تطابق الكلمات المفتاحية
    query_words = set(query.lower().split())
    
    for result in results:
        content_words = set(result["content"].lower().split())
        word_overlap = len(query_words.intersection(content_words))
        result["keyword_score"] = word_overlap / len(query_words) if query_words else 0
        
        # درجة مركبة
        result["final_score"] = (
            result["relevance_score"] * 0.7 + 
            result["keyword_score"] * 0.3
        )
    
    # ترتيب حسب الدرجة المركبة
    return sorted(results, key=lambda x: x["final_score"], reverse=True)

# --- 6. وكيل الإجابة المتقدم ---
def advanced_research_agent(query: str, search_depth: int) -> Dict:
    """وكيل بحثي متقدم مع تحليل شامل."""
    if not model:
        return {
            "answer": "❌ النظام غير مهيأ (API KEY مفقود أو غير صالح).",
            "confidence": 0,
            "sources": [],
            "category": "error"
        }

    # معالجة التحيات البسيطة
    q_lower = query.lower().strip()
    if any(word in q_lower for word in ["سلام", "مرحبا", "اهلا", "هاي", "شكرا"]):
        return {
            "answer": "🤗 وعليكم السلام ومرحباً بك! أنا العرّاب البحثي المحسّن، جاهز للإجابة على جميع أسئلتك حول علم وراثة الحمام بأسلوب علمي متقدم.",
            "confidence": 1.0,
            "sources": [],
            "category": "greeting"
        }

    # تصنيف السؤال
    category = classify_question(query)
    
    # البحث المتقدم
    with st.spinner("🔬 جارٍ البحث المتقدم في قاعدة المعرفة..."):
        search_results = search_knowledge_advanced(query, category, top_k=search_depth)
    
    if not search_results:
        return {
            "answer": "🤔 لم أجد معلومات مباشرة متعلقة بهذا السؤال في قاعدة المعرفة الحالية. جرب إعادة صياغة السؤال أو استخدام مصطلحات أخرى.",
            "confidence": 0,
            "sources": [],
            "category": category
        }

    # بناء السياق مع درجات الصلة
    context_parts = []
    source_info = []
    
    for i, result in enumerate(search_results):
        context_parts.append(f"[مرجع {i+1} - درجة الصلة: {result['final_score']:.2f}]\n{result['content']}")
        source_info.append({
            "rank": i+1,
            "score": result['final_score'],
            "content_preview": result['content'][:100] + "..."
        })
    
    context_text = "\n\n" + "="*50 + "\n\n".join(context_parts)

    # بناء البرومبت المتقدم
    prompt = create_advanced_prompt(query, context_text, category)
    
    # توليد الإجابة
    with st.spinner("🧠 جارٍ تحليل المعلومات وصياغة الإجابة المتخصصة..."):
        try:
            response = model.generate_content(prompt)
            answer = response.text
            
            # تقدير مستوى الثقة
            confidence = estimate_confidence(search_results, answer)
            
            return {
                "answer": answer,
                "confidence": confidence,
                "sources": source_info,
                "category": category,
                "search_results": search_results
            }
            
        except Exception as e:
            return {
                "answer": f"❌ حدث خطأ في توليد الإجابة: {str(e)}",
                "confidence": 0,
                "sources": source_info,
                "category": category
            }

def create_advanced_prompt(query: str, context: str, category: str) -> str:
    """إنشاء برومبت متقدم حسب فئة السؤال."""
    
    base_prompt = f"""
أنت "العرّاب الذكي V2.0"، خبير عالمي في علم وراثة الحمام وتربيته، مزود بقدرات تحليلية متقدمة.

**فئة السؤال المحددة:** {category}
**السؤال:** {query}

**المراجع العلمية المرتبة حسب الصلة:**
{context}

**مهامك المتخصصة:**

1. **التحليل العلمي الدقيق:** قدم إجابة علمية شاملة ومدعومة بالأدلة من المراجع
2. **التصنيف والتنظيم:** رتب المعلومات بشكل منطقي مع عناوين واضحة
3. **الربط والاستنتاج:** اربط المفاهيم المختلفة وقدم استنتاجات منطقية
4. **التطبيق العملي:** اذكر التطبيقات العملية عند الإمكان

**معايير الجودة:**
✅ استخدم المعلومات من المراجع حصرياً
✅ أسلوب علمي واضح ومنظم
✅ تدرج من العام إلى التفصيلي
✅ أمثلة عملية عند الحاجة
✅ اذكر أي قيود في المعلومات المتاحة

**تنسيق الإجابة المطلوب:**
استخدم العناوين (##) والنقاط والجداول عند الحاجة لتحسين القابلية للقراءة.
"""

    # تخصيصات إضافية حسب الفئة
    category_specific = {
        "genetic_inheritance": "\n**تركيز خاص:** اشرح آليات الوراثة والجينات بالتفصيل مع الأمثلة.",
        "breeding": "\n**تركيز خاص:** ركز على الجوانب العملية للتربية والتزاوج والانتقاء.",
        "colors": "\n**تركيز خاص:** فصل وراثة الألوان والطفرات اللونية بشكل مفصل.",
        "diseases": "\n**تركيز خاص:** اشرح الأمراض وأساليب الوقاية والعلاج.",
        "analysis": "\n**تركيز خاص:** قدم تحليلاً عميقاً ومقارنات وتفسيرات شاملة."
    }
    
    return base_prompt + category_specific.get(category, "")

def estimate_confidence(search_results: List[Dict], answer: str) -> float:
    """تقدير مستوى الثقة في الإجابة."""
    if not search_results:
        return 0.0
    
    # عوامل الثقة
    avg_relevance = np.mean([r['final_score'] for r in search_results])
    num_sources = len(search_results)
    answer_length = len(answer.split())
    
    # حساب درجة الثقة
    confidence = (
        avg_relevance * 0.5 +
        min(num_sources / 5, 1.0) * 0.3 +
        min(answer_length / 200, 1.0) * 0.2
    )
    
    return min(confidence, 1.0)

# --- 7. واجهة المستخدم المحسّنة ---
def main():
    # العنوان الرئيسي
    st.markdown("""
    # 🚀 العرّاب للجينات V2.0
    ## وكيل بحثي ذكي متقدم لعلم وراثة الحمام
    ---
    """)

    # الشريط الجانبي المحسّن
    settings = create_enhanced_sidebar()
    
    # القسم الرئيسي للمحادثة
    create_chat_interface(settings)
    
    # إحصائيات وتحليلات
    if st.checkbox("📊 عرض إحصائيات الجلسة"):
        show_session_statistics()

def create_enhanced_sidebar():
    """إنشاء شريط جانبي محسّن مع معلومات تفصيلية."""
    st.sidebar.markdown("## 📊 حالة النظام المتقدمة")
    
    # حالة قاعدة المعرفة
    if vector_db and embedder:
        st.sidebar.success("✅ قاعدة المعرفة جاهزة")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("🧠 مقاطع معرفية", len(vector_db['chunks']))
        with col2:
            st.metric("📁 البيانات الوصفية", len(metadata) if metadata else 0)
            
        # معلومات إضافية
        if metadata:
            st.sidebar.info(f"📅 آخر تحديث: {metadata.get('last_updated', 'غير محدد')}")
            
    else:
        st.sidebar.error("❌ قاعدة المعرفة غير متوفرة")
    
    # حالة النموذج
    if model:
        st.sidebar.success("✅ المساعد الذكي جاهز")
        st.sidebar.info("🤖 النموذج: Gemini 1.5 Flash")
    else:
        st.sidebar.error("❌ المساعد الذكي غير متاح")
    
    # إعدادات البحث
    st.sidebar.markdown("## ⚙️ إعدادات البحث")
    search_depth = st.sidebar.slider("عمق البحث", 3, 10, 5)
    
    return {"search_depth": search_depth}

def create_chat_interface(settings: Dict):
    """إنشاء واجهة المحادثة المحسّنة."""
    st.markdown("### 💬 محادثة ذكية مع العرّاب")
    
    # تهيئة الجلسة
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_stats" not in st.session_state:
        st.session_state.session_stats = {
            "questions_asked": 0,
            "categories_used": set(),
            "avg_confidence": 0.0,
            "start_time": datetime.now()
        }

    # عرض المحادثات السابقة
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "metadata" in message:
                display_enhanced_response(message["content"], message["metadata"])
            else:
                st.markdown(message["content"])

    # إدخال السؤال الجديد
    if prompt := st.chat_input("🧬 اسألني أي شيء عن وراثة الحمام..."):
        # إضافة سؤال المستخدم
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # إنشاء الإجابة
        with st.chat_message("assistant"):
            response_data = advanced_research_agent(prompt, settings["search_depth"])
            
            # عرض الإجابة المحسّنة
            display_enhanced_response(response_data["answer"], response_data)
            
            # حفظ الإجابة مع البيانات الوصفية
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_data["answer"],
                "metadata": response_data
            })
            
            # تحديث الإحصائيات
            update_session_stats(response_data)
        
        # إعادة تشغيل الواجهة لتحديث الإحصائيات
        st.rerun()

def display_enhanced_response(answer: str, metadata: dict):
    """عرض الإجابة مع المعلومات الإضافية."""
    # الإجابة الرئيسية
    st.markdown(answer)
    
    # معلومات إضافية
    if metadata.get("confidence", 0) > 0:
        # شريط الثقة
        confidence = metadata["confidence"]
        confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
        
        st.markdown(f"""
        <div style="margin: 10px 0;">
            <strong>🎯 مستوى الثقة:</strong>
            <div style="background-color: #f0f0f0; border-radius: 10px; padding: 5px;">
                <div style="background-color: {confidence_color}; width: {confidence*100}%; height: 20px; border-radius: 10px; text-align: center; color: white; line-height: 20px;">
                    {confidence:.1%}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # معلومات المصادر
        if metadata.get("sources") and len(metadata["sources"]) > 0:
            with st.expander(f"📚 المصادر المستخدمة ({len(metadata['sources'])})"):
                for i, source in enumerate(metadata["sources"]):
                    st.markdown(f"""
                    **مصدر {source['rank']}** (درجة الصلة: {source['score']:.2f})
                    {source['content_preview']}
                    """)

def update_session_stats(response_data: dict):
    """تحديث إحصائيات الجلسة."""
    stats = st.session_state.session_stats
    stats["questions_asked"] += 1
    stats["categories_used"].add(response_data.get("category", "unknown"))
    
    if response_data.get("confidence", 0) > 0:
        # تحديث متوسط الثقة
        current_avg = stats["avg_confidence"]
        new_avg = (current_avg * (stats["questions_asked"] - 1) + response_data["confidence"]) / stats["questions_asked"]
        stats["avg_confidence"] = new_avg

def show_session_statistics():
    """عرض إحصائيات الجلسة."""
    stats = st.session_state.session_stats
    
    st.markdown("### 📈 إحصائيات الجلسة الحالية")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🤔 الأسئلة المطروحة", stats["questions_asked"])
    
    with col2:
        st.metric("🎯 متوسط الثقة", f"{stats['avg_confidence']:.1%}")
    
    with col3:
        st.metric("📂 الفئات المستخدمة", len(stats["categories_used"]))
    
    with col4:
        duration = datetime.now() - stats["start_time"]
        st.metric("⏱️ مدة الجلسة", f"{duration.seconds//60} دقيقة")
    
    # رسم بياني للفئات
    if len(stats["categories_used"]) > 0:
        categories_list = list(stats["categories_used"])
        fig = px.pie(
            values=[1] * len(categories_list), 
            names=categories_list,
            title="توزيع فئات الأسئلة"
        )
        st.plotly_chart(fig, use_container_width=True)

# --- 8. تشغيل التطبيق ---
if __name__ == "__main__":
    main()
