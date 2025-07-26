# ===================================================================
# 🚀 العرّاب للجينات V5.0 - النسخة الاحترافية
# دمج تصميمك المتقدم مع المحركات البرمجية الكاملة
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
    page_title="العرّاب للجينات V5.0",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# --- 2. CSS مخصص للواجهة ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. قواعد البيانات ---
GENE_DATA = {
    'B': { 'display_name_ar': "اللون الأساسي", 'type_en': 'sex-linked', 'alleles': { 'BA': {'name': 'آش ريد'}, '+': {'name': 'أزرق/أسود'}, 'b': {'name': 'بني'} }, 'dominance': ['BA', '+', 'b'] },
    'd': { 'display_name_ar': "التخفيف", 'type_en': 'sex-linked', 'alleles': { '+': {'name': 'عادي (غير مخفف)'}, 'd': {'name': 'مخفف'} }, 'dominance': ['+', 'd'] },
    'e': { 'display_name_ar': "أحمر متنحي", 'type_en': 'autosomal', 'alleles': { '+': {'name': 'عادي (غير أحمر متنحي)'}, 'e': {'name': 'أحمر متنحي'} }, 'dominance': ['+', 'e'] },
    'C': { 'display_name_ar': "النمط", 'type_en': 'autosomal', 'alleles': { 'CT': {'name': 'نمط تي (مخملي)'}, 'C': {'name': 'تشيكر'}, '+': {'name': 'بار (شريط)'}, 'c': {'name': 'بدون شريط'} }, 'dominance': ['CT', 'C', '+', 'c'] },
    'S': { 'display_name_ar': "الانتشار (سبريد)", 'type_en': 'autosomal', 'alleles': { 'S': {'name': 'منتشر (سبريد)'}, '+': {'name': 'عادي (غير منتشر)'} }, 'dominance': ['S', '+'] }
}
GENE_ORDER = list(GENE_DATA.keys())
NAME_TO_SYMBOL_MAP = {
    gene: {info['name']: symbol for symbol, info in data['alleles'].items()}
    for gene, data in GENE_DATA.items()
}
TRUSTED_SOURCES = {
    'جينات وسلالات': [{'name': "Ronald Huntley's Pigeon Genetics", 'url': 'http://www.huntley.pigeonwebsite.com/'}, {'name': "Pigeon Genetics Center", 'url': 'http://www.pigen.org/'}],
    'جمعيات رسمية': [{'name': 'National Pigeon Association (NPA)', 'url': 'https://www.npausa.com/'}, {'name': 'American Racing Pigeon Union (AU)', 'url': 'https://www.pigeon.org/'}],
    'صحة وعلاجات': [{'name': 'Merck Veterinary Manual - Pigeons', 'url': 'https://www.merckvetmanual.com/poultry/pigeons-and-doves'}],
    'مصادر علمية': [{'name': 'PubMed (Pigeon Genetics)', 'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=pigeon+genetics'}]
}

# --- 4. إدارة الجلسة ---
def initialize_session_state():
    defaults = {
        "messages": [],
        "session_stats": {"queries": 0, "deep_searches": 0, "live_searches": 0, "calculations": 0}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- 5. تحميل الموارد ---
@st.cache_resource
def load_resources():
    resources = {"vector_db": None, "embedder": None, "model": None}
    if VECTOR_SEARCH_AVAILABLE:
        vector_db_path = "vector_db.pkl"
        if os.path.exists(vector_db_path):
            try:
                with open(vector_db_path, "rb") as f:
                    resources["vector_db"] = pickle.load(f)
                resources["embedder"] = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            except Exception as e:
                st.error(f"خطأ في تحميل قاعدة المتجهات: {e}")
    
    if GEMINI_AVAILABLE and "GEMINI_API_KEY" in st.secrets:
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            resources["model"] = genai.GenerativeModel('gemini-1.5-flash', generation_config={"temperature": 0.1, "max_output_tokens": 3000})
        except Exception as e:
            st.error(f"فشل تهيئة Gemini: {e}")
    return resources

# --- 6. وظائف الوكيل الذكي ---
def search_deep_memory(query: str, resources: dict, top_k: int = 5) -> List[Dict]:
    """البحث في الذاكرة العميقة (قاعدة المتجهات)."""
    if not resources.get("vector_db") or not resources.get("embedder"): return []
    index = resources["vector_db"]["index"]
    chunks = resources["vector_db"]["chunks"]
    metadata = resources["vector_db"].get("metadata", [])
    query_embedding = resources["embedder"].encode([query])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    return [{"type": "gene", "source": m.get('source', 'قاعدة البيانات المحلية'), "content": c, "relevance": 1 / (1 + d), "metadata": m} for d, i in zip(distances[0], indices[0]) if i < len(chunks) for c, m in [(chunks[i], metadata[i] if i < len(metadata) else {})]]

def get_live_memory_links(query: str) -> List[Dict]:
    """اقتراح روابط من الذاكرة الحية (المصادر الموثوقة)."""
    links_to_add = []
    query_lower = query.lower()
    if any(k in query_lower for k in ["health", "disease", "treatment", "صحة", "مرض", "علاج"]):
        links_to_add.extend(TRUSTED_SOURCES['صحة وعلاجات'])
    if any(k in query_lower for k in ["npa", "au", "if", "جمعية", "اتحاد"]):
        links_to_add.extend(TRUSTED_SOURCES['جمعيات رسمية'])
    if any(k in query_lower for k in ["research", "study", "pubmed", "science", "بحث", "دراسة"]):
        links_to_add.extend(TRUSTED_SOURCES['مصادر علمية'])
    
    unique_links = {link['url']: link for link in links_to_add}.values()
    return list(unique_links)

def process_message(query: str, resources: dict):
    """المعالج الرئيسي للأسئلة."""
    st.session_state.session_stats["queries"] += 1
    
    if not resources.get("model"):
        return {"answer": "❌ نظام الذكاء الاصطناعي غير مهيأ. يرجى التحقق من مفتاح API.", "sources": []}

    deep_results = search_deep_memory(query, resources)
    if deep_results:
        st.session_state.session_stats["deep_searches"] += 1
        
    live_links = get_live_memory_links(query)
    if live_links:
        st.session_state.session_stats["live_searches"] += 1

    context = "\n\n---\n\n".join([f"مصدر داخلي: {r['content']}" for r in deep_results])
    
    prompt = f"""
    أنت 'العرّاب للجينات V5.0'، خبير عالمي في وراثة الحمام.
    أجب على السؤال التالي '{query}' بناءً على السياق الداخلي.
    إذا كانت المعلومات غير كافية، وضح ذلك.

    السياق الداخلي:
    {context}

    الإجابة التحليلية:
    """
    
    try:
        response = resources["model"].generate_content(prompt)
        answer = response.text
        return {"answer": answer, "sources": deep_results, "links": live_links}
    except Exception as e:
        return {"answer": f"❌ خطأ: {str(e)}", "sources": deep_results, "links": live_links}

# --- 7. الحاسبة الوراثية ---
class GeneticCalculator:
    def describe_phenotype(self, genotype_dict):
        phenotypes = {gene: "" for gene in GENE_ORDER}
        for gene_name, gt_part in genotype_dict.items():
            alleles = gt_part.replace('•//', '').split('//')
            for dominant_allele in GENE_DATA[gene_name]['dominance']:
                if dominant_allele in alleles:
                    phenotypes[gene_name] = GENE_DATA[gene_name]['alleles'][dominant_allele]['name']
                    break
        if 'e//e' in genotype_dict.get('e', ''): phenotypes['B'] = 'أحمر متنحي'; phenotypes['C'] = ''
        if 'S' in genotype_dict.get('S', ''):
            if 'e//e' not in genotype_dict.get('e', ''): phenotypes['C'] = 'منتشر (سبريد)'
        sex = "أنثى" if any('•' in genotype_dict.get(g, '') for g, d in GENE_DATA.items() if d['type_en'] == 'sex-linked') else "ذكر"
        desc_parts = [phenotypes.get('B')]
        if phenotypes.get('C'): desc_parts.append(phenotypes.get('C'))
        gt_str_parts = [genotype_dict[gene].strip() for gene in GENE_ORDER]
        gt_str = " | ".join(gt_str_parts)
        final_phenotype = " ".join(filter(None, desc_parts))
        return f"{sex} {final_phenotype}", gt_str

def predict_genetics_final(parent_inputs):
    # ... (الكود الكامل لهذه الوظيفة موجود في النسخ السابقة، تم إخفاؤه هنا للاختصار)
    calculator = GeneticCalculator()
    parent_genotypes = {}
    for parent in ['male', 'female']:
        gt_parts = []
        for gene in GENE_ORDER:
            gene_info = GENE_DATA[gene]
            visible_name = parent_inputs[parent].get(f'{gene}_visible')
            hidden_name = parent_inputs[parent].get(f'{gene}_hidden', visible_name)
            wild_type_symbol = next((s for s, n in gene_info['alleles'].items() if '+' in s or '⁺' in s), gene_info['dominance'][0])
            visible_symbol = NAME_TO_SYMBOL_MAP[gene].get(visible_name, wild_type_symbol)
            hidden_symbol = NAME_TO_SYMBOL_MAP[gene].get(hidden_name, visible_symbol)
            if gene_info['type_en'] == 'sex-linked' and parent == 'female':
                gt_parts.append(f"•//{visible_symbol}")
            else:
                alleles = sorted([visible_symbol, hidden_symbol], key=lambda x: gene_info['dominance'].index(x))
                gt_parts.append(f"{alleles[0]}//{alleles[1]}")
        parent_genotypes[parent] = gt_parts
    def generate_gametes(genotype_parts, is_female):
        parts_for_product = []
        for i, gt_part in enumerate(genotype_parts):
            gene_name = GENE_ORDER[i]
            if GENE_DATA[gene_name]['type_en'] == 'sex-linked' and is_female:
                parts_for_product.append([gt_part.replace('•//','').strip()])
            else:
                parts_for_product.append(gt_part.split('//'))
        return list(product(*parts_for_product))
    male_gametes = generate_gametes(parent_genotypes['male'], is_female=False)
    female_gametes = generate_gametes(parent_genotypes['female'], is_female=True)
    offspring_counts = collections.Counter()
    for m_gamete in male_gametes:
        for f_gamete in female_gametes:
            son_dict, daughter_dict = {}, {}
            for i, gene in enumerate(GENE_ORDER):
                alleles = sorted([m_gamete[i], f_gamete[i]], key=lambda x: GENE_DATA[gene]['dominance'].index(x))
                if GENE_DATA[gene]['type_en'] == 'sex-linked':
                    son_dict[gene], daughter_dict[gene] = f"{alleles[0]}//{alleles[1]}", f"•//{m_gamete[i]}"
                else:
                    gt_part = f"{alleles[0]}//{alleles[1]}"
                    son_dict[gene], daughter_dict[gene] = gt_part, gt_part
            offspring_counts[calculator.describe_phenotype(son_dict)] += 1
            offspring_counts[calculator.describe_phenotype(daughter_dict)] += 1
    return offspring_counts

# --- 8. واجهة المستخدم الرئيسية ---
def main():
    """الواجهة الرئيسية للتطبيق."""
    initialize_session_state()
    resources = load_resources()
    
    st.markdown('<div class="main-header"><h1>🚀 العرّاب للجينات V5.0</h1><p>النسخة الاحترافية - تحليل متقدم بالذكاء الاصطناعي</p></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["💬 المحادثة الذكية", "🧬 الحاسبة الوراثية", "📊 التحليلات"])

    with tab1:
        st.header("💬 تحدث مع الخبير")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and message.get("links"):
                    st.markdown("**🌐 مراجع خارجية موصى بها:**")
                    for link in message["links"]:
                        st.markdown(f"- [{link['name']}]({link['url']})")

        if prompt := st.chat_input("اسألني أي شيء عن وراثة الحمام..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                response_data = process_message(prompt, resources)
                st.markdown(response_data["answer"])
                if response_data["links"]:
                    st.markdown("**🌐 مراجع خارجية موصى بها:**")
                    for link in response_data["links"]:
                        st.markdown(f"- [{link['name']}]({link['url']})")
                st.session_state.messages.append({"role": "assistant", "content": response_data["answer"], "links": response_data["links"]})
            st.rerun()

    with tab2:
        st.header("🧮 الحاسبة الوراثية التفاعلية")
        parent_inputs = {'male': {}, 'female': {}}
        input_col, result_col = st.columns([2, 3])
        
        with input_col:
            st.subheader("📝 إدخال بيانات الوالدين")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**♂️ الذكر (الأب)**")
                for gene, data in GENE_DATA.items():
                    with st.expander(f"{data['display_name_ar']}"):
                        choices = ["(لا اختيار)"] + [v['name'] for v in data['alleles'].values()]
                        parent_inputs['male'][f'{gene}_visible'] = st.selectbox("الصفة الظاهرية", choices, key=f"male_{gene}_visible")
                        parent_inputs['male'][f'{gene}_hidden'] = st.selectbox("الصفة الخفية", choices, key=f"male_{gene}_hidden")
            with col2:
                st.markdown("**♀️ الأنثى (الأم)**")
                for gene, data in GENE_DATA.items():
                    with st.expander(f"{data['display_name_ar']}"):
                        choices = ["(لا اختيار)"] + [v['name'] for v in data['alleles'].values()]
                        parent_inputs['female'][f'{gene}_visible'] = st.selectbox("الصفة الظاهرية", choices, key=f"female_{gene}_visible")
                        if data['type_en'] != 'sex-linked':
                            parent_inputs['female'][f'{gene}_hidden'] = st.selectbox("الصفة الخفية", choices, key=f"female_{gene}_hidden")
                        else:
                            st.info("لا يوجد صفة خفية")
                            parent_inputs['female'][f'{gene}_hidden'] = parent_inputs['female'][f'{gene}_visible']
        
        with result_col:
            st.subheader("📊 النتائج المتوقعة")
            if st.button("⚡ احسب الآن", use_container_width=True, type="primary"):
                if not all([parent_inputs['male'].get('B_visible') != "(لا اختيار)", parent_inputs['female'].get('B_visible') != "(لا اختيار)"]):
                    st.error("⚠️ الرجاء اختيار اللون الأساسي لكلا الوالدين.")
                else:
                    with st.spinner("🧮 جاري الحساب..."):
                        results = predict_genetics_final(parent_inputs)
                        total = sum(results.values())
                        st.success(f"✅ تم حساب {total} تركيبة!")
                        st.session_state.session_stats['calculations'] += 1
                        df_results = pd.DataFrame([{'النمط الظاهري': p, 'النمط الوراثي': g, 'النسبة %': f"{(c/total)*100:.1f}%"} for (p, g), c in results.items()])
                        st.dataframe(df_results, use_container_width=True)
                        chart_data = df_results.set_index('النمط الظاهري')['النسبة %'].str.rstrip('%').astype('float')
                        st.bar_chart(chart_data)

    with tab3:
        st.header("📊 تحليلات الجلسة")
        stats = st.session_state.session_stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("الاستعلامات", stats["queries"])
        col2.metric("بحث عميق", stats["deep_searches"])
        col3.metric("بحث حي", stats["live_searches"])
        col4.metric("حسابات", stats["calculations"])

# --- 9. تشغيل التطبيق ---
if __name__ == "__main__":
    main()
