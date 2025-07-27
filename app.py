# ===================================================================
# 🧬 العرّاب للجينات v6.1 - وكيل الذكاء الاصطناعي المتقدم
# ===================================================================

import streamlit as st
from itertools import product
import collections
import pandas as pd
import numpy as np
import pickle
import os
import re
from datetime import datetime
from typing import list, dict, tuple
import time

# --- التحقق من توفر المكتبات ---
try:
    import google.generativeai as genai
    gemini_available = True
except ImportError:
    gemini_available = False

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    vector_search_available = True
except ImportError:
    vector_search_available = False

# --- إعدادات الصفحة ---
st.set_page_config(
    layout="wide",
    page_title="العرّاب للجينات v6.1",
    page_icon="🧬",
    initial_sidebar_state="collapsed"
)

# --- CSS متقدم للواجهة ---
st.markdown("""
<style>
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .custom-container {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f5;
        margin: 20px auto;
        max-width: 1200px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }

    .header-bar {
        background: linear-gradient(90deg, #4facfe, #00f2fe);
        color: white;
        padding: 20px;
        border-radius: 10px 10px 0 0;
        text-align: center;
    }

    .button {
        margin-top: 10px;
    }

    .chat-area {
        background-color: #fff;
        border-radius: 10px;
        padding: 20px;
        height: 60vh;
        overflow-y: auto;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }

    .input-area {
        padding: 20px;
        border-top: 1px solid #e9ecef;
        background: #f8f9fa;
    }

    .result-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- قواعد البيانات المحسنة ---
gene_data = {
    'b': {
        'display_name_ar': "اللون الأساسي", 
        'type_en': 'sex-linked', 
        'emoji': '🎨',
        'alleles': {
            'ba': {'name': 'آش ريد'},
            '+': {'name': 'أزرق/أسود'},
            'b': {'name': 'بني'}
        },
        'dominance': ['ba', '+', 'b']
    },
    'd': {
        'display_name_ar': "التخفيف", 
        'type_en': 'sex-linked', 
        'emoji': '💧',
        'alleles': {
            '+': {'name': 'عادي (غير مخفف)'},
            'd': {'name': 'مخفف'}
        },
        'dominance': ['+', 'd']
    },
    'e': {
        'display_name_ar': "أحمر متنحي", 
        'type_en': 'autosomal', 
        'emoji': '🔴',
        'alleles': {
            '+': {'name': 'عادي (غير أحمر متنحي)'},
            'e': {'name': 'أحمر متنحي'}
        },
        'dominance': ['+', 'e']
    },
    'c': {
        'display_name_ar': "النمط", 
        'type_en': 'autosomal', 
        'emoji': '📐',
        'alleles': {
            'ct': {'name': 'نمط تي (مخملي)'},
            'c': {'name': 'تشيكر'},
            '+': {'name': 'بار (شريط)'},
            'c': {'name': 'بدون شريط'}
        },
        'dominance': ['ct', 'c', '+', 'c']
    },
    's': {
        'display_name_ar': "الانتشار (سبريد)", 
        'type_en': 'autosomal', 
        'emoji': '🌊',
        'alleles': {
            's': {'name': 'منتشر (سبريد)'},
            '+': {'name': 'عادي (غير منتشر)'}
        },
        'dominance': ['s', '+']
    }
}
gene_order = list(gene_data.keys())
name_to_symbol_map = {
    gene: {info['name']: symbol for symbol, info in data['alleles'].items()}
    for gene, data in gene_data.items()
}

# --- إدارة الجلسة المحسنة ---
def initialize_session_state():
    """تهيئة متغيرات الجلسة."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None

# --- تحميل الموارد المحسّن ---
@st.cache_resource
def load_resources():
    """تحميل جميع الموارد اللازمة للوكيل الذكي."""
    resources = {"status": "limited"}

    if vector_search_available:
        vector_db_path = "pigeon_knowledge_base_v8.0.pkl"
        if os.path.exists(vector_db_path):
            try:
                with open(vector_db_path, "rb") as f:
                    resources["vector_db"] = pickle.load(f)
                resources["embedder"] = SentenceTransformer("sentence-transformers/paraphrase-multilingual-minilm-l12-v2")
            except Exception as e:
                st.warning(f"⚠️ تعذر تحميل قاعدة المتجهات: {e}")

    if gemini_available and "gemini_api_key" in st.secrets:
        try:
            genai.configure(api_key=st.secrets["gemini_api_key"])
            resources["model"] = genai.generativemodel('gemini-1.5-flash',
                generation_config={"temperature": 0.1, "max_output_tokens": 3000})
            resources["status"] = "ready"
        except Exception as e:
            st.error(f"❌ فشل تفعيل الذكاء الاصطناعي: {e}")
            resources["status"] = "error"

    return resources

# --- فئة الحاسبة الوراثية المحسنة ---
class AdvancedGeneticCalculator:
    def describe_phenotype(self, genotype_dict: dict[str, str]) -> tuple[str, str]:
        phenotypes = {gene: "" for gene in gene_order}
        for gene_name, gt_part in genotype_dict.items():
            alleles = gt_part.replace('•//', '').split('//')
            for dominant_allele in gene_data[gene_name]['dominance']:
                if dominant_allele in alleles:
                    phenotypes[gene_name] = gene_data[gene_name]['alleles'][dominant_allele]['name']
                    break

        if 'e//e' in genotype_dict.get('e', ''):
            phenotypes['b'] = 'أحمر متنحي'
            phenotypes['c'] = ''

        if 's' in genotype_dict.get('s', ''):
            if 'e//e' not in genotype_dict.get('e', ''):
                phenotypes['c'] = 'منتشر (سبريد)'

        sex = "أنثى" if any('•' in genotype_dict.get(g, '') for g, d in gene_data.items() if d['type_en'] == 'sex-linked') else "ذكر"

        desc_parts = [phenotypes.get('b')]
        if phenotypes.get('d') == 'مخفف': desc_parts.append('مخفف')
        if phenotypes.get('c'): desc_parts.append(phenotypes.get('c'))

        gt_str_parts = [genotype_dict[gene].strip() for gene in gene_order]
        gt_str = " | ".join(gt_str_parts)
        final_phenotype = " ".join(filter(None, desc_parts))
        return f"{sex} {final_phenotype}", gt_str

    def calculate_advanced_genetics(self, parent_inputs: dict) -> dict:
        try:
            parent_genotypes = {}
            for parent in ['male', 'female']:
                gt_parts = []
                for gene in gene_order:
                    gene_info = gene_data[gene]
                    visible_name = parent_inputs[parent].get(f'{gene}_visible')
                    hidden_name = parent_inputs[parent].get(f'{gene}_hidden', visible_name)
                    wild_type_symbol = next((s for s, n in gene_info['alleles'].items() if '+' in s), gene_info['dominance'][0])
                    visible_symbol = name_to_symbol_map[gene].get(visible_name, wild_type_symbol)
                    hidden_symbol = name_to_symbol_map[gene].get(hidden_name, visible_symbol)

                    if gene_info['type_en'] == 'sex-linked' and parent == 'female':
                        gt_parts.append(f"•//{visible_symbol}")
                    else:
                        alleles = sorted([visible_symbol, hidden_symbol], key=lambda x: gene_info['dominance'].index(x))
                        gt_parts.append(f"{alleles[0]}//{alleles[1]}")
                parent_genotypes[parent] = gt_parts

            def generate_gametes(genotype_parts, is_female):
                parts_for_product = []
                for i, gt_part in enumerate(genotype_parts):
                    gene_name = gene_order[i]
                    if gene_data[gene_name]['type_en'] == 'sex-linked' and is_female:
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
                    for i, gene in enumerate(gene_order):
                        alleles = sorted([m_gamete[i], f_gamete[i]], key=lambda x: gene_data[gene]['dominance'].index(x))
                        if gene_data[gene]['type_en'] == 'sex-linked':
                            son_dict[gene] = f"{alleles[0]}//{alleles[1]}"
                            daughter_dict[gene] = f"•//{m_gamete[i]}"
                        else:
                            gt_part = f"{alleles[0]}//{alleles[1]}"
                            son_dict[gene] = gt_part
                            daughter_dict[gene] = gt_part
                    offspring_counts[self.describe_phenotype(son_dict)] += 1
                    offspring_counts[self.describe_phenotype(daughter_dict)] += 1

            total = sum(offspring_counts.values())
            return {'results': offspring_counts, 'total': total}
        except Exception as e:
            return {'error': f"خطأ في الحساب: {str(e)}"}

# --- الوكيل الذكي المتقدم ---
class IntelligentGeneticAgent:
    def __init__(self, resources: dict):
        self.resources = resources
        self.calculator = AdvancedGeneticCalculator()

    def understand_query(self, query: str) -> dict:
        intent = {'type': 'general', 'calculation_needed': False}
        if any(keyword in query.lower() for keyword in ['احسب', 'حساب', 'نتائج', 'تزاوج', 'تربية']):
            intent['type'] = 'calculation'
            intent['calculation_needed'] = True
        elif any(keyword in query.lower() for keyword in ['شرح', 'وضح', 'كيف', 'ماذا', 'لماذا']):
            intent['type'] = 'explanation'
        elif any(keyword in query.lower() for keyword in ['مساعدة', 'help', 'مرحبا', 'السلام']):
            intent['type'] = 'greeting'
        return intent

    def search_deep_memory(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.resources.get("vector_db") or not self.resources.get("embedder"): return []
        try:
            index = self.resources["vector_db"]["index"]
            chunks = self.resources["vector_db"]["chunks"]
            query_embedding = self.resources["embedder"].encode([query])
            distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
            return [{"content": chunks[idx], "relevance": 1 / (1 + dist)} for dist, idx in zip(distances[0], indices[0]) if idx < len(chunks)]
        except Exception:
            return []

    def generate_smart_response(self, query: str, intent: dict) -> dict:
        if not self.resources.get("model"):
            return {"answer": "❌ عذراً، نظام الذكاء الاصطناعي غير متاح حالياً.", "calculation_widget": intent['calculation_needed']}

        deep_results = self.search_deep_memory(query)
        context = "\n\n".join([f"معلومة: {r['content']}" for r in deep_results[:3]])

        system_prompt = "أنت 'العرّاب للجينات v6.1'، وكيل ذكاء اصطناعي متخصص في وراثة الحمام..."
        user_prompt = f"سؤال: {query}\nالسياق: {context}"

        try:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.resources["model"].generate_content(full_prompt)
            return {"answer": response.text, "sources": deep_results, "calculation_widget": intent['calculation_needed']}
        except Exception as e:
            return {"answer": f"❌ عذراً، حدث خطأ: {str(e)}", "sources": deep_results, "calculation_widget": intent['calculation_needed']}

# --- الحاسبة المدمجة العصرية ---
def render_embedded_calculator():
    st.markdown('<div class="result-card"><h3>🧮 الحاسبة الوراثية المدمجة</h3></div>', unsafe_allow_html=True)
    with st.container():
        col1, col2 = st.columns(2)
        parent_inputs = {'male': {}, 'female': {}}

        with col1:
            st.markdown("#### ♂️ **الذكر (الأب)**")
            for gene, data in gene_data.items():
                with st.expander(f"{data['emoji']} {data['display_name_ar']}"):
                    choices = ["(اختر الصفة)"] + [v['name'] for v in data['alleles'].values()]
                    parent_inputs['male'][f'{gene}_visible'] = st.selectbox("الصفة الظاهرة:", choices, key=f"emb_male_{gene}_visible")
                    parent_inputs['male'][f'{gene}_hidden'] = st.selectbox("الصفة المخفية:", choices, key=f"emb_male_{gene}_hidden")

        with col2:
            st.markdown("#### ♀️ **الأنثى (الأم)**")
            for gene, data in gene_data.items():
                with st.expander(f"{data['emoji']} {data['display_name_ar']}"):
                    choices = ["(اختر الصفة)"] + [v['name'] for v in data['alleles'].values()]
                    parent_inputs['female'][f'{gene}_visible'] = st.selectbox("الصفة الظاهرة:", choices, key=f"emb_female_{gene}_visible")
                    if data['type_en'] != 'sex-linked':
                        parent_inputs['female'][f'{gene}_hidden'] = st.selectbox("الصفة المخفية:", choices, key=f"emb_female_{gene}_hidden")
                    else:
                        st.info("الإناث لديها أليل واحد فقط")
                        parent_inputs['female'][f'{gene}_hidden'] = parent_inputs['female'][f'{gene}_visible']

        if st.button("🚀 احسب النتائج المتوقعة", use_container_width=True, type="primary"):
            if not all([parent_inputs['male'].get('b_visible') != "(اختر الصفة)", parent_inputs['female'].get('b_visible') != "(اختر الصفة)"]):
                st.error("⚠️ يرجى اختيار اللون الأساسي لكلا الوالدين")
            else:
                with st.spinner("🧬 جاري الحساب..."):
                    calculator = AdvancedGeneticCalculator()
                    result_data = calculator.calculate_advanced_genetics(parent_inputs)
                    if 'error' in result_data:
                        st.error(result_data['error'])
                    else:
                        display_advanced_results(result_data)

def display_advanced_results(result_data: dict):
    st.markdown('<div class="result-card"><h3>📊 النتائج المتوقعة</h3></div>', unsafe_allow_html=True)
    results = result_data['results']
    total = result_data['total']
    df_results = pd.DataFrame([{'النمط الظاهري': p, 'النمط الوراثي': g, 'العدد': c, 'النسبة %': f"{(c/total)*100:.1f}%"} for (p, g), c in results.items()])
    st.dataframe(df_results, use_container_width=True, hide_index=True)
    chart_data = df_results.set_index('النمط الظاهري')['النسبة %'].str.rstrip('%').astype('float')
    st.bar_chart(chart_data, height=300)

# --- الواجهة الرئيسية ---
def main():
    initialize_session_state()

    if 'agent' not in st.session_state or st.session_state.agent is None:
        resources = load_resources()
        st.session_state.agent = IntelligentGeneticAgent(resources)

    agent = st.session_state.agent

    # إضافة رسالة ترحيب إذا لم تكن موجودة
    if not st.session_state.messages:
        welcome_message = "🧬 **مرحباً بك في العرّاب للجينات v6.1!** أنا وكيلك الذكي المتخصص. كيف يمكنني مساعدتك؟"
        st.session_state.messages.append({"role": "assistant", "content": welcome_message, "show_calculator": False})

    # حاوية الواجهة الرئيسية
    st.markdown('<div class="custom-container">', unsafe_allow_html=True)

    # شريط العنوان
    st.markdown(f'''
    <div class="header-bar">
        🧬 العرّاب للجينات v6.1
        <div style="font-size: 14px; opacity: 0.9;">
            وكيل ذكي متقدم • {agent.resources['status']}
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # منطقة المحادثة
    chat_area = st.container()
    with chat_area:
        st.markdown('<div class="chat-area">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="message user-message"><div class="user-bubble">{message["content"]}</div></div>', unsafe_allow_html=True)
            else:  # assistant
                st.markdown(f'<div class="message assistant-message"><div class="avatar">🤖</div><div class="assistant-bubble">{message["content"]}</div></div>', unsafe_allow_html=True)
                if message.get("show_calculator"):
                    render_embedded_calculator()

        if st.session_state.get('typing_indicator'):
            st.markdown('<div class="message assistant-message"><div class="avatar">🤖</div><div class="typing-indicator"><span>العرّاب يفكر...</span></div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # منطقة الإدخال
    st.markdown('<div class="input-area">', unsafe_allow_html=True)

    # حقل الإدخال الرئيسي
    user_input = st.chat_input("اكتب سؤالك هنا... 💬", key="main_input")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.typing_indicator = True
        st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True)

    # معالجة الرسالة الأخيرة
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and st.session_state.typing_indicator:
        last_message = st.session_state.messages[-1]["content"]

        intent = agent.understand_query(last_message)
        response_data = agent.generate_smart_response(last_message, intent)

        assistant_message = {
            "role": "assistant",
            "content": response_data["answer"],
            "sources": response_data.get("sources", []),
            "show_calculator": response_data.get("calculation_widget", False),
        }
        st.session_state.messages.append(assistant_message)
        st.session_state.typing_indicator = False
        st.rerun()

if __name__ == "__main__":
    main()
