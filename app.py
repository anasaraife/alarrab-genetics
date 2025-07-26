# ===================================================================
# 🚀 العرّاب للجينات V2.1 - دمج الحاسبة الوراثية
# تم إعادة دمج الحاسبة الوراثية التفاعلية مع الوكيل البحثي المتقدم
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
    page_title="العرّاب للجينات V2.1",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# --- 2. قاعدة البيانات الوراثية (للحاسبة والوكيل) ---
GENE_DATA = {
    'B': {
        'display_name_ar': "اللون الأساسي", 'type_en': 'sex-linked',
        'alleles': { 'BA': {'name': 'آش ريد'}, '+': {'name': 'أزرق/أسود'}, 'b': {'name': 'بني'} },
        'dominance': ['BA', '+', 'b']
    },
    'd': {
        'display_name_ar': "التخفيف", 'type_en': 'sex-linked',
        'alleles': { '+': {'name': 'عادي (غير مخفف)'}, 'd': {'name': 'مخفف'} },
        'dominance': ['+', 'd']
    },
    'e': {
        'display_name_ar': "أحمر متنحي", 'type_en': 'autosomal',
        'alleles': { '+': {'name': 'عادي (غير أحمر متنحي)'}, 'e': {'name': 'أحمر متنحي'} },
        'dominance': ['+', 'e']
    },
    'C': {
        'display_name_ar': "النمط", 'type_en': 'autosomal',
        'alleles': { 'CT': {'name': 'نمط تي (مخملي)'}, 'C': {'name': 'تشيكر'}, '+': {'name': 'بار (شريط)'}, 'c': {'name': 'بدون شريط'} },
        'dominance': ['CT', 'C', '+', 'c']
    },
    'S': {
        'display_name_ar': "الانتشار (سبريد)", 'type_en': 'autosomal',
        'alleles': { 'S': {'name': 'منتشر (سبريد)'}, '+': {'name': 'عادي (غير منتشر)'} },
        'dominance': ['S', '+']
    }
}
GENE_ORDER = list(GENE_DATA.keys())
NAME_TO_SYMBOL_MAP = {
    gene: {info['name']: symbol for symbol, info in data['alleles'].items()}
    for gene, data in GENE_DATA.items()
}

# --- 3. تحميل الموارد ---
@st.cache_resource
def load_resources():
    # ... (الكود كما هو في النسخة السابقة) ...
    resources = { "vector_db": None, "embedder": None, "metadata": None }
    if VECTOR_SEARCH_AVAILABLE:
        vector_db_path = "vector_db.pkl"
        metadata_path = "vector_metadata.json"
        if os.path.exists(vector_db_path):
            try:
                with open(vector_db_path, "rb") as f:
                    resources["vector_db"] = pickle.load(f)
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        resources["metadata"] = json.load(f)
                resources["embedder"] = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            except Exception as e:
                st.error(f"خطأ في تحميل قاعدة المتجهات: {e}")
    return resources

@st.cache_resource
def initialize_gemini():
    # ... (الكود كما هو في النسخة السابقة) ...
    if "GEMINI_API_KEY" in st.secrets:
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            return genai.GenerativeModel('gemini-1.5-flash', generation_config={"temperature": 0.1, "max_output_tokens": 3000})
        except Exception as e:
            st.error(f"فشل تهيئة Gemini: {e}")
    return None

# تحميل الموارد
resources = load_resources()
model = initialize_gemini()

# --- 4. المحرك الوراثي (للحاسبة) ---
class GeneticCalculator:
    def describe_phenotype(self, genotype_dict):
        phenotypes = {gene: "" for gene in GENE_ORDER}
        for gene_name, gt_part in genotype_dict.items():
            alleles = gt_part.replace('•//', '').split('//')
            for dominant_allele in GENE_DATA[gene_name]['dominance']:
                if dominant_allele in alleles:
                    phenotypes[gene_name] = GENE_DATA[gene_name]['alleles'][dominant_allele]['name']
                    break
        if 'e//e' in genotype_dict.get('e', ''):
            phenotypes['B'] = 'أحمر متنحي'; phenotypes['C'] = ''
        if 'S' in genotype_dict.get('S', ''):
            if 'e//e' not in genotype_dict.get('e', ''):
                phenotypes['C'] = 'منتشر (سبريد)'
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

# --- 5. وظائف الوكيل البحثي (كما هي) ---
def search_knowledge_advanced(query: str, top_k: int = 5) -> List[Dict]:
    # ... (الكود كما هو في النسخة السابقة) ...
    if not resources["vector_db"] or not resources["embedder"]: return []
    index = resources["vector_db"]["index"]
    chunks = resources["vector_db"]["chunks"]
    query_embedding = resources["embedder"].encode([query])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    return [{"content": chunks[idx], "score": 1 / (1 + dist)} for dist, idx in zip(distances[0], indices[0]) if idx < len(chunks)]

def advanced_research_agent(query: str) -> Dict:
    # ... (الكود كما هو في النسخة السابقة) ...
    if not model: return {"answer": "❌ النظام غير مهيأ.", "confidence": 0, "sources": []}
    q_lower = query.lower().strip()
    if any(word in q_lower for word in ["سلام", "مرحبا", "اهلا"]): return {"answer": "🤗 وعليكم السلام!", "confidence": 1.0, "sources": []}
    search_results = search_knowledge_advanced(query)
    if not search_results: return {"answer": "🤔 لم أجد معلومات مباشرة.", "confidence": 0, "sources": []}
    context_text = "\n\n---\n\n".join([r['content'] for r in search_results])
    prompt = f"أنت خبير وراثة. أجب على السؤال '{query}' بناءً على السياق التالي فقط:\n\n{context_text}\n\nالإجابة التحليلية:"
    try:
        response = model.generate_content(prompt)
        answer = response.text
        confidence = np.mean([r['score'] for r in search_results]) if search_results else 0
        return {"answer": answer, "confidence": confidence, "sources": search_results}
    except Exception as e:
        return {"answer": f"❌ خطأ: {str(e)}", "confidence": 0, "sources": search_results}

# --- 6. واجهة المستخدم الرئيسية ---
def main():
    """الواجهة الرئيسية للتطبيق."""
    st.markdown(" # 🚀 العرّاب للجينات V2.1 (نسخة الاختبار)")
    
    # استخدام علامات التبويب للفصل بين الوظائف
    tab1, tab2 = st.tabs(["🤖 المحادثة البحثية", "🧬 الحاسبة الوراثية"])

    with tab1:
        st.header("💬 تحدث مع الوكيل البحثي")
        if "messages" not in st.session_state: st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("اسألني أي شيء عن وراثة الحمام..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                response_data = advanced_research_agent(prompt)
                st.markdown(response_data["answer"])
                # يمكن إضافة عرض المصادر ومستوى الثقة هنا
            st.session_state.messages.append({"role": "assistant", "content": response_data["answer"]})
            st.rerun()

    with tab2:
        st.header("🧮 الحاسبة الوراثية التفاعلية")
        parent_inputs = {'male': {}, 'female': {}}
        input_col, result_col = st.columns([2, 3])
        
        with input_col:
            st.subheader("📝 إدخال البيانات")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**♂️ الذكر (الأب)**")
                for gene, data in GENE_DATA.items():
                    with st.expander(f"{data['display_name_ar']}"):
                        choices = ["(لا اختيار)"] + [v['name'] for v in data['alleles'].values()]
                        parent_inputs['male'][f'{gene}_visible'] = st.selectbox("الصفة الظاهرية", choices, key=f"male_{gene}_visible")
                        parent_inputs['male'][f'{gene}_hidden'] = st.selectbox("الصفة الخفية", choices, key=f"male_{gene}_hidden")
            with col2:
                st.write("**♀️ الأنثى (الأم)**")
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
                        df_results = pd.DataFrame([{'النمط الظاهري': p, 'النمط الوراثي': g, 'النسبة %': f"{(c/total)*100:.1f}%"} for (p, g), c in results.items()])
                        st.dataframe(df_results, use_container_width=True)
                        chart_data = df_results.set_index('النمط الظاهري')['النسبة %'].str.rstrip('%').astype('float')
                        st.bar_chart(chart_data)

# --- 7. تشغيل التطبيق ---
if __name__ == "__main__":
    main()
