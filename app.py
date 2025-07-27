# ===================================================================
# 🚀 العرّاب للجينات V5.0 - النسخة الشاملة النهائية
# دمج جميع المميزات: واجهة متقدمة + حاسبة وراثية + مصادر موثوقة + تحليل بصري
# تصميم وهيكلة: أنس العرايفة | دمج وتكييف: Gemini
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
from typing import List, Dict, Tuple
import plotly.express as px

# --- التحقق من توفر المكتبات المطلوبة ---
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

# --- 1. إعدادات الصفحة المحسنة ---
st.set_page_config(
    layout="wide",
    page_title="العرّاب للجينات V5.0",
    page_icon="🧬",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': """
        # العرّاب للجينات V5.0
        النسخة الشاملة النهائية - أداة متكاملة لتحليل وراثة وألوان الحمام
        """
    }
)

# --- 2. CSS مخصص للواجهة ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .info-box {
        background: linear-gradient(135deg, #f0f2f6 0%, #e8f5e8 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    
    .calculator-section {
        background: linear-gradient(135deg, #f8f9ff 0%, #e6f3ff 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #e0e6ed;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(255,255,255,0.1);
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 0 30px;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .trusted-source {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. قواعد البيانات الوراثية والمصادر ---
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

TRUSTED_SOURCES = {
    'جينات وسلالات': [
        {'name': 'Pigeon Breeding: Genetics At Work', 'url': 'https://www.amazon.com/Pigeon-Breeding-Genetics-Work/dp/1888963098', 'description': 'كتاب شامل عن وراثة الحمام'},
        {'name': "Ronald Huntley's Pigeon Genetics", 'url': 'http://www.huntley.pigeonwebsite.com/', 'description': 'موقع رونالد هانتلي للوراثة'},
    ],
    'جمعيات رسمية': [
        {'name': 'National Pigeon Association (NPA)', 'url': 'https://www.npausa.com/', 'description': 'الجمعية الوطنية الأمريكية'},
    ],
    'صحة وعلاجات': [
        {'name': 'Merck Veterinary Manual', 'url': 'https://www.merckvetmanual.com/poultry/pigeons-and-doves', 'description': 'دليل الطب البيطري'},
    ],
    'مصادر علمية': [
        {'name': 'PubMed (Pigeon Genetics)', 'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=pigeon+genetics', 'description': 'قاعدة البيانات العلمية'},
    ]
}

# --- 4. إدارة الجلسة المحسنة ---
def initialize_session_state():
    """تهيئة حالة الجلسة مع إعدادات شاملة."""
    defaults = {
        "messages": [],
        "search_history": [],
        "calculation_history": [],
        "user_preferences": {
            "max_results": 10,
            "analysis_depth": "متوسط",
            "language_style": "علمي",
            "include_charts": True,
            "show_trusted_sources": True,
        },
        "session_stats": {
            "queries_count": 0,
            "successful_searches": 0,
            "charts_generated": 0,
            "calculations_performed": 0,
            "sources_referenced": 0
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- 5. تحميل الموارد المتقدم ---
@st.cache_resource(show_spinner="جاري تحميل الموارد الأساسية...")
def load_enhanced_resources():
    """تحميل جميع الموارد المطلوبة مع معالجة شاملة للأخطاء."""
    resources = {"status": "loading"}
    
    if VECTOR_SEARCH_AVAILABLE:
        vector_db_path = "pigeon_knowledge_base_v8.0.pkl"
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
    """تهيئة نموذج Gemini مع إعدادات محسنة."""
    if not GEMINI_AVAILABLE: return None
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key: return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash',
            generation_config={"temperature": 0.1, "max_output_tokens": 4096})
        return model
    except Exception as e:
        st.error(f"فشل تهيئة Gemini: {e}")
        return None

# --- 6. المحرك الوراثي المطور ---
class EnhancedGeneticCalculator:
    def describe_phenotype(self, genotype_dict):
        phenotypes = {gene: "" for gene in GENE_ORDER}
        for gene_name, gt_part in genotype_dict.items():
            alleles = gt_part.replace('•//', '').split('//')
            for dominant_allele in GENE_DATA[gene_name]['dominance']:
                if dominant_allele in alleles:
                    phenotypes[gene_name] = GENE_DATA[gene_name]['alleles'][dominant_allele]['name']
                    break
        
        if 'e//e' in genotype_dict.get('e', ''):
            phenotypes['B'] = 'أحمر متنحي'
            phenotypes['C'] = ''
        
        if 'S' in genotype_dict.get('S', ''):
            if 'e//e' not in genotype_dict.get('e', ''):
                phenotypes['C'] = 'منتشر (سبريد)'
        
        sex = "أنثى" if any('•' in genotype_dict.get(g, '') for g, d in GENE_DATA.items() if d['type_en'] == 'sex-linked') else "ذكر"
        
        desc_parts = [phenotypes.get('B')]
        if phenotypes.get('d') == 'مخفف': desc_parts.append('مخفف')
        if phenotypes.get('C'): desc_parts.append(phenotypes.get('C'))
        
        gt_str_parts = [genotype_dict[gene].strip() for gene in GENE_ORDER]
        gt_str = " | ".join(gt_str_parts)
        final_phenotype = " ".join(filter(None, desc_parts))
        return f"{sex} {final_phenotype}", gt_str

    def calculate_detailed_genetics(self, parent_inputs):
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
                            son_dict[gene] = f"{alleles[0]}//{alleles[1]}"
                            daughter_dict[gene] = f"•//{m_gamete[i]}"
                        else:
                            gt_part = f"{alleles[0]}//{alleles[1]}"
                            son_dict[gene] = gt_part
                            daughter_dict[gene] = gt_part
                    offspring_counts[self.describe_phenotype(son_dict)] += 1
                    offspring_counts[self.describe_phenotype(daughter_dict)] += 1
            
            total_offspring = sum(offspring_counts.values())
            sex_dist = {'ذكر': sum(c for (p,g),c in offspring_counts.items() if 'ذكر' in p), 'أنثى': sum(c for (p,g),c in offspring_counts.items() if 'أنثى' in p)}
            
            return {
                'results': offspring_counts,
                'total_offspring': total_offspring,
                'sex_distribution': sex_dist,
            }
        except Exception as e:
            return {'error': f"خطأ في الحساب: {str(e)}"}

# --- 7. البحث والتحليل المتقدم ---
def enhanced_search_knowledge(query: str, resources: dict, top_k: int = 5) -> List[Dict]:
    if not resources.get("vector_db") or not resources.get("embedder"):
        return []
    try:
        index = resources["vector_db"]["index"]
        chunks = resources["vector_db"]["chunks"]
        query_embedding = resources["embedder"].encode([query])
        distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
        return [{"content": chunks[idx], "score": 1 / (1 + dist)} for dist, idx in zip(distances[0], indices[0]) if idx < len(chunks)]
    except Exception as e:
        st.warning(f"خطأ في البحث الدلالي: {e}")
        return []

def get_relevant_trusted_sources(query: str) -> List[Dict]:
    relevant_sources = []
    # ... (منطق تحديد المصادر الموثوقة)
    return relevant_sources

# --- 8. الوكيل الخبير المتطور ---
def ultimate_expert_agent(query: str, resources: dict, preferences: dict) -> Dict:
    st.session_state.session_stats["queries_count"] += 1
    
    if not resources.get("model"):
        # Fallback response if AI model is not available
        return {"answer": "❌ نظام الذكاء الاصطناعي غير متاح. يرجى إعداد مفتاح API.", "confidence": 0.1}

    with st.spinner("🔍 البحث في قاعدة المعرفة..."):
        search_results = enhanced_search_knowledge(query, resources, top_k=preferences.get("max_results", 5))

    context_text = "\n\n---\n\n".join([r['content'] for r in search_results])
    
    prompt = f"""
أنت "العرّاب V5.0 - الخبير الشامل".
أجب على سؤال المستخدم بدقة وعلمية بالاعتماد على السياق التالي.

السياق:
{context_text}

سؤال المستخدم:
{query}

التحليل الشامل:
"""
    with st.spinner("🧠 التحليل الاستنتاجي المتقدم..."):
        try:
            ai_response = resources["model"].generate_content(prompt)
            final_answer = ai_response.text
            confidence = np.mean([r['score'] for r in search_results]) if search_results else 0.3
            return {"answer": final_answer, "confidence": confidence, "sources": search_results}
        except Exception as e:
            return {"answer": f"❌ خطأ في التحليل: {str(e)}", "confidence": 0.2, "sources": search_results}

# --- 9. واجهة المستخدم الشاملة ---
def main():
    initialize_session_state()
    resources = load_enhanced_resources()
    model = initialize_enhanced_gemini()
    resources["model"] = model

    st.markdown('<div class="main-header"><h1>🚀 العرّاب للجينات V5.0</h1><p><strong>النسخة الشاملة النهائية</strong></p></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["💬 المحادثة الذكية", "🧬 الحاسبة الوراثية", "⚙️ الإعدادات والتاريخ"])

    with tab1:
        st.subheader("🤖 تحدث مع الخبير الوراثي")
        chat_container = st.container(height=500)
        for message in st.session_state.messages:
            with chat_container.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("مثال: ما هي الجينات المسؤولة عن اللون البني؟"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            chat_container.chat_message("user").markdown(prompt)
            
            with chat_container.chat_message("assistant"):
                response_data = ultimate_expert_agent(prompt, resources, st.session_state.user_preferences)
                st.markdown(response_data["answer"])
                st.session_state.messages.append({"role": "assistant", "content": response_data["answer"]})

    with tab2:
        st.subheader("🧮 الحاسبة الوراثية المتقدمة")
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
                    calculator = EnhancedGeneticCalculator()
                    result_data = calculator.calculate_detailed_genetics(parent_inputs)
                    st.session_state.calculation_history.append(result_data)
                    st.session_state.session_stats["calculations_performed"] += 1

        # عرض آخر نتيجة حساب
        if st.session_state.calculation_history:
            last_calc = st.session_state.calculation_history[-1]
            st.subheader("📊 أحدث النتائج")
            if 'error' in last_calc:
                st.error(last_calc['error'])
            else:
                df_results = pd.DataFrame([{'النمط الظاهري': p, 'النمط الوراثي': g, 'النسبة %': f"{(c/last_calc['total_offspring'])*100:.1f}%"} for (p, g), c in last_calc['results'].items()])
                st.dataframe(df_results, use_container_width=True)
                
                fig = px.pie(values=list(last_calc['color_distribution'].values()), names=list(last_calc['color_distribution'].keys()), title="توزيع الألوان")
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("⚙️ الإعدادات وتاريخ الجلسة")
        st.markdown("---")
        with st.expander("🔧 إعدادات البحث والتحليل"):
            prefs = st.session_state.user_preferences
            prefs['max_results'] = st.slider("عدد نتائج البحث", 5, 20, prefs['max_results'])
            prefs['analysis_depth'] = st.select_slider("عمق التحليل", ["بسيط", "متوسط", "عميق"], value=prefs['analysis_depth'])
            prefs['language_style'] = st.radio("أسلوب اللغة", ["علمي", "مبسط", "تقني"], index=["علمي", "مبسط", "تقني"].index(prefs['language_style']))
            prefs['include_charts'] = st.toggle("تضمين الرسوم البيانية", value=prefs['include_charts'])
        
        with st.expander("📜 تاريخ الحسابات"):
            if st.session_state.calculation_history:
                for i, calc in enumerate(reversed(st.session_state.calculation_history)):
                    st.json(calc, expanded=False)
            else:
                st.info("لا يوجد تاريخ حسابات بعد.")

if __name__ == "__main__":
    main()
