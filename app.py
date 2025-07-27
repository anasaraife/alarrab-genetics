# ===================================================================
# 🚀 العرّاب للجينات V5.4 - النسخة المستقرة والنهائية
# تم إعادة هيكلة الكود بالكامل لضمان الاستقرار والأداء العالي.
# ===================================================================

import streamlit as st
from itertools import product
import collections
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from typing import List, Dict, Tuple
import plotly.express as px

# --- التحقق المبدئي من توفر المكتبات ---
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
    page_title="العرّاب للجينات V5.4",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# --- 2. قواعد البيانات الوراثية ---
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

# --- 3. إدارة الجلسة ---
def initialize_session_state():
    """تهيئة حالة الجلسة."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "calculation_history" not in st.session_state:
        st.session_state.calculation_history = []
    if "resources_loaded" not in st.session_state:
        st.session_state.resources_loaded = False
    if "resources" not in st.session_state:
        st.session_state.resources = {}
    if "model" not in st.session_state:
        st.session_state.model = None


# --- 4. تحميل الموارد (عند الحاجة فقط) ---
def load_resources():
    """تحميل الموارد الثقيلة عند الحاجة وتخزينها في الجلسة."""
    st.session_state.resources_loaded = True
    resources = {"status": "loading"}
    
    with st.spinner("🧠 جاري تحميل عقل الوكيل..."):
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
    
    st.session_state.resources = resources
    
    with st.spinner("🔑 جاري تهيئة الذكاء الاصطناعي..."):
        if GEMINI_AVAILABLE:
            api_key = st.secrets.get("GEMINI_API_KEY")
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-1.5-flash',
                        generation_config={"temperature": 0.2, "max_output_tokens": 4096})
                    st.session_state.model = model
                except Exception as e:
                    st.error(f"فشل تهيئة Gemini: {e}")
            else:
                st.warning("مفتاح Gemini API غير موجود في الأسرار.")
        else:
            st.warning("مكتبة Gemini غير متاحة.")

# --- 5. المحرك الوراثي ---
class GeneticCalculator:
    def describe_phenotype(self, genotype_dict):
        phenotypes = {gene: "" for gene in GENE_ORDER}
        for gene_name, gt_part in genotype_dict.items():
            alleles = gt_part.replace('•//', '').split('//')
            for dominant_allele in GENE_DATA[gene_name]['dominance']:
                if dominant_allele in alleles:
                    phenotypes[gene_name] = GENE_DATA[gene_name]['alleles'][dominant_allele]['name']
                    break
        
        if 'e//e' in genotype_dict.get('e', ''): phenotypes['B'], phenotypes['C'] = 'أحمر متنحي', ''
        if 'S' in genotype_dict.get('S', '') and 'e//e' not in genotype_dict.get('e', ''): phenotypes['C'] = 'منتشر (سبريد)'
        
        sex = "أنثى" if any('•' in genotype_dict.get(g, '') for g, d in GENE_DATA.items() if d['type_en'] == 'sex-linked') else "ذكر"
        
        desc_parts = [phenotypes.get('B')]
        if phenotypes.get('d') == 'مخفف': desc_parts.append('مخفف')
        if phenotypes.get('C'): desc_parts.append(phenotypes.get('C'))
        
        gt_str = " | ".join([genotype_dict[gene].strip() for gene in GENE_ORDER])
        final_phenotype = " ".join(filter(None, desc_parts))
        return f"{sex} {final_phenotype}", gt_str

    def calculate_genetics(self, parent_inputs):
        try:
            parent_genotypes = {}
            for parent in ['male', 'female']:
                gt_parts = []
                for gene in GENE_ORDER:
                    gene_info = GENE_DATA[gene]
                    visible_name = parent_inputs[parent].get(f'{gene}_visible')
                    hidden_name = parent_inputs[parent].get(f'{gene}_hidden', visible_name)
                    wild_type_symbol = next((s for s, n in gene_info['alleles'].items() if '+' in s), gene_info['dominance'][0])
                    visible_symbol = NAME_TO_SYMBOL_MAP[gene].get(visible_name, wild_type_symbol)
                    hidden_symbol = NAME_TO_SYMBOL_MAP[gene].get(hidden_name, wild_type_symbol)
                    
                    if gene_info['type_en'] == 'sex-linked' and parent == 'female':
                        gt_parts.append(f"•//{visible_symbol}")
                    else:
                        alleles = sorted([visible_symbol, hidden_symbol], key=lambda x: gene_info['dominance'].index(x))
                        gt_parts.append(f"{alleles[0]}//{alleles[1]}")
                parent_genotypes[parent] = gt_parts

            def generate_gametes(genotype_parts, is_female):
                parts = []
                for i, gt_part in enumerate(genotype_parts):
                    gene_name = GENE_ORDER[i]
                    if GENE_DATA[gene_name]['type_en'] == 'sex-linked' and is_female:
                        parts.append([gt_part.replace('•//','').strip()])
                    else:
                        parts.append(gt_part.split('//'))
                return list(product(*parts))

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
                    offspring_counts[self.describe_phenotype(son_dict)] += 1
                    offspring_counts[self.describe_phenotype(daughter_dict)] += 1
            
            return {'results': offspring_counts, 'total_offspring': sum(offspring_counts.values())}
        except Exception as e:
            return {'error': f"خطأ في الحساب: {str(e)}"}

# --- 6. الوكيل الذكي ---
class ExpertAgent:
    def __init__(self, resources, model):
        self.resources = resources
        self.model = model

    def search_knowledge(self, query: str, top_k: int = 5) -> str:
        if not self.resources.get("vector_db") or not self.resources.get("embedder"):
            return "قاعدة المعرفة غير متاحة حالياً."
        try:
            index = self.resources["vector_db"]["index"]
            chunks = self.resources["vector_db"]["chunks"]
            query_embedding = self.resources["embedder"].encode([query])
            _, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
            return "\n\n---\n\n".join([chunks[idx] for idx in indices[0] if idx < len(chunks)])
        except Exception as e:
            return f"خطأ في البحث: {e}"

    def process_query(self, query: str) -> str:
        if not self.model:
            return "❌ نظام الذكاء الاصطناعي غير متاح. يرجى إعداد مفتاح API."

        context = self.search_knowledge(query)
        prompt = f"أنت خبير في وراثة الحمام. أجب على السؤال التالي بالاعتماد على السياق.\n\nالسياق:\n{context}\n\nالسؤال:\n{query}\n\nالإجابة:"
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ خطأ في التحليل: {str(e)}"

# --- 7. واجهة المستخدم ---
def main():
    initialize_session_state()

    st.title("🚀 العرّاب للجينات V5.4 - النسخة المستقرة")

    # زر تحميل الموارد إذا لم يتم تحميلها
    if not st.session_state.resources_loaded:
        st.info("يحتاج التطبيق إلى تحميل الموارد الأساسية للبدء.")
        if st.button("🚀 بدء وتحميل الموارد"):
            load_resources()
            st.rerun()
        return # إيقاف التنفيذ حتى يتم تحميل الموارد

    agent = ExpertAgent(st.session_state.resources, st.session_state.model)
    
    tab1, tab2 = st.tabs(["💬 المحادثة الذكية", "🧬 الحاسبة الوراثية"])

    with tab1:
        st.subheader("🤖 تحدث مع الخبير الوراثي")
        chat_container = st.container(height=500, border=True)
        for message in st.session_state.messages:
            with chat_container.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("اسأل عن وراثة الحمام..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            chat_container.chat_message("user").markdown(prompt)
            
            with chat_container.chat_message("assistant"):
                with st.spinner("🧠 العرّاب يفكر..."):
                    response = agent.process_query(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    with tab2:
        st.subheader("🧮 الحاسبة الوراثية")
        with st.container(border=True):
            col1, col2 = st.columns(2)
            parent_inputs = {'male': {}, 'female': {}}
            
            with col1:
                st.markdown("#### ♂️ **الذكر (الأب)**")
                for gene, data in GENE_DATA.items():
                    choices = ["(اختر)"] + [v['name'] for v in data['alleles'].values()]
                    parent_inputs['male'][f'{gene}_visible'] = st.selectbox(f"**{data['display_name_ar']}** (الظاهر):", choices, key=f"calc_male_{gene}_visible")
                    parent_inputs['male'][f'{gene}_hidden'] = st.selectbox(f"**{data['display_name_ar']}** (الخفي):", choices, key=f"calc_male_{gene}_hidden")
            
            with col2:
                st.markdown("#### ♀️ **الأنثى (الأم)**")
                for gene, data in GENE_DATA.items():
                    choices = ["(اختر)"] + [v['name'] for v in data['alleles'].values()]
                    parent_inputs['female'][f'{gene}_visible'] = st.selectbox(f"**{data['display_name_ar']}** (الظاهر):", choices, key=f"calc_female_{gene}_visible")
                    if data['type_en'] != 'sex-linked':
                        parent_inputs['female'][f'{gene}_hidden'] = st.selectbox(f"**{data['display_name_ar']}** (الخفي):", choices, key=f"calc_female_{gene}_hidden")
                    else:
                        st.info(f"**{data['display_name_ar']}**: الإناث لديها أليل واحد فقط.", icon="ℹ️")
                        parent_inputs['female'][f'{gene}_hidden'] = parent_inputs['female'][f'{gene}_visible']

            if st.button("🚀 احسب النتائج", use_container_width=True, type="primary"):
                if not all(val != "(اختر)" for val in [parent_inputs['male']['B_visible'], parent_inputs['female']['B_visible']]):
                    st.error("⚠️ يرجى اختيار اللون الأساسي لكلا الوالدين.")
                else:
                    calculator = GeneticCalculator()
                    result_data = calculator.calculate_genetics(parent_inputs)
                    st.session_state.calculation_history.append(result_data)

        if st.session_state.calculation_history:
            last_calc = st.session_state.calculation_history[-1]
            st.subheader("📊 أحدث النتائج")
            if 'error' in last_calc:
                st.error(last_calc['error'])
            else:
                df_results = pd.DataFrame([{'النمط الظاهري': p, 'النمط الوراثي': g, 'النسبة %': f"{(c/last_calc['total_offspring'])*100:.1f}%"} for (p, g), c in last_calc['results'].items()])
                st.dataframe(df_results, use_container_width=True)
                chart_data = df_results.set_index('النمط الظاهري')['النسبة %'].str.rstrip('%').astype('float')
                st.bar_chart(chart_data)

if __name__ == "__main__":
    main()
