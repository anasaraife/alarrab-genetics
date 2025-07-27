# ===================================================================
# 🚀 العرّاب للجينات V5.2 - وكيل ذكي بشخصية متطورة
# تطوير جذري في الوكيل الذكي للمحادثة والتحليل العلمي
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
    page_title="العرّاب للجينات V5.2",
    page_icon="🧬",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': """
        # العرّاب للجينات V5.2
        النسخة الذكية المتطورة - وكيل ذكي بشخصية علمية محسنة
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
    
    .agent-thinking {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 0.5rem 0;
        font-style: italic;
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

# --- 4. إدارة الجلسة المحسنة ---
def initialize_session_state():
    """تهيئة حالة الجلسة مع إعدادات شاملة."""
    defaults = {
        "messages": [],
        "search_history": [],
        "calculation_history": [],
        "conversation_context": [],
        "agent_memory": {},
        "user_preferences": {
            "max_results": 10,
            "analysis_depth": "متوسط",
            "language_style": "علمي",
            "include_charts": True,
            "show_trusted_sources": True,
            "conversation_mode": "تفاعلي",
            "thinking_visibility": True,
        },
        "session_stats": {
            "queries_count": 0,
            "successful_searches": 0,
            "charts_generated": 0,
            "calculations_performed": 0,
            "sources_referenced": 0,
            "deep_analyses": 0,
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
    """تهيئة نموذج Gemini مع إعدادات محسنة."""
    if not GEMINI_AVAILABLE: return None
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key: return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash',
            generation_config={"temperature": 0.2, "max_output_tokens": 6000})
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

# --- 8. فئة الوكيل الذكي المتطور الجديد ---
class IntelligentGeneticsAgent:
    def __init__(self, resources: dict, preferences: dict):
        self.resources = resources
        self.preferences = preferences

    def analyze_query_intent(self, query: str) -> Dict:
        intent_analysis = {"type": "معلوماتي", "requires_calculation": False}
        if any(word in query.lower() for word in ["احسب", "حساب", "نسبة", "احتمال"]):
            intent_analysis["type"] = "حسابي"
            intent_analysis["requires_calculation"] = True
        return intent_analysis

    def generate_thinking_process(self, query: str, intent: Dict, search_results: List) -> str:
        thinking_steps = []
        if intent["type"] == "حسابي":
            thinking_steps.append("🧮 أتعرف على متطلبات الحساب الوراثي")
            thinking_steps.append("📊 سأطبق قوانين مندل وأسس الوراثة")
        else:
            thinking_steps.append("💭 أحلل استفسارك وأحدد أفضل طريقة للإجابة")
            if search_results:
                thinking_steps.append(f"📚 راجعت {len(search_results)} مرجع من مكتبتي")
        return " → ".join(thinking_steps)

    def create_advanced_prompt(self, query: str, context: str, intent: Dict) -> str:
        return f"""
أنت **د. العرّاب الوراثي** - خبير متخصص في علم الوراثة وألوان الحمام.
مهمتك هي الإجابة على سؤال المستخدم بدقة وعلمية.

**السياق المرجعي من مكتبتك:**
```
{context}
```
**سؤال المستخدم الحالي:**
{query}

**تعليمات:**
- التزم بما هو موجود في السياق المرجعي.
- إذا لم تجد المعلومة، قل ذلك بوضوح.
- كن متفهماً ومساعداً في أسلوبك.

**ابدأ إجابتك الآن:**
"""

    def process_query(self, query: str) -> Dict:
        st.session_state.session_stats["queries_count"] += 1
        
        if not self.resources.get("model"):
            return {"answer": "❌ نظام الذكاء الاصطناعي غير متاح حالياً."}

        intent = self.analyze_query_intent(query)
        search_results = enhanced_search_knowledge(query, self.resources, top_k=self.preferences.get("max_results", 8))
        thinking_process = self.generate_thinking_process(query, intent, search_results)
        
        context_text = "\n\n---\n\n".join([r['content'] for r in search_results])
        advanced_prompt = self.create_advanced_prompt(query, context_text, intent)
        
        try:
            ai_response = self.resources["model"].generate_content(advanced_prompt)
            final_answer = ai_response.text
            return {"answer": final_answer, "thinking": thinking_process, "sources": search_results, "intent": intent}
        except Exception as e:
            return {"answer": f"❌ عذراً، واجهت صعوبة: {str(e)}", "thinking": "❌ حدث خطأ أثناء التحليل"}

# --- 9. واجهة المستخدم الشاملة ---
def main():
    initialize_session_state()
    resources = load_enhanced_resources()
    model = initialize_enhanced_gemini()
    resources["model"] = model
    
    agent = IntelligentGeneticsAgent(resources, st.session_state.user_preferences)

    st.markdown('<div class="main-header"><h1>🚀 العرّاب للجينات V5.2</h1><p><strong>النسخة الذكية المتطورة</strong></p></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["💬 المحادثة الذكية", "🧬 الحاسبة الوراثية", "⚙️ الإعدادات"])

    with tab1:
        st.subheader("🤖 تحدث مع د. العرّاب الوراثي")
        chat_container = st.container(height=500)
        
        for message in st.session_state.messages:
            with chat_container.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and st.session_state.user_preferences["thinking_visibility"] and message.get("thinking"):
                    with st.expander("🧠 كيف فكرت في هذا؟"):
                        st.markdown(f'<div class="agent-thinking">💭 {message["thinking"]}</div>', unsafe_allow_html=True)

        if prompt := st.chat_input("مثال: ما هي مصادرك؟"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with chat_container.chat_message("user"):
                st.markdown(prompt)
            
            with chat_container.chat_message("assistant"):
                with st.spinner("🤔 د. العرّاب يفكر..."):
                    response_data = agent.process_query(prompt)
                
                st.markdown(response_data["answer"])
                
                if st.session_state.user_preferences["thinking_visibility"] and response_data.get("thinking"):
                    with st.expander("🧠 كيف فكرت في هذا؟"):
                        st.markdown(f'<div class="agent-thinking">💭 {response_data["thinking"]}</div>', unsafe_allow_html=True)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_data["answer"],
                    "thinking": response_data.get("thinking", "")
                })

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

        if st.session_state.calculation_history:
            last_calc = st.session_state.calculation_history[-1]
            st.subheader("📊 أحدث النتائج")
            if 'error' in last_calc:
                st.error(last_calc['error'])
            else:
                df_results = pd.DataFrame([{'النمط الظاهري': p, 'النمط الوراثي': g, 'النسبة %': f"{(c/last_calc['total_offspring'])*100:.1f}%"} for (p, g), c in last_calc['results'].items()])
                st.dataframe(df_results, use_container_width=True)
                
                fig = px.pie(values=list(last_calc['sex_distribution'].values()), names=list(last_calc['sex_distribution'].keys()), title="توزيع الجنس")
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("⚙️ الإعدادات وتاريخ الجلسة")
        with st.expander("🔧 إعدادات البحث والتحليل"):
            prefs = st.session_state.user_preferences
            prefs['thinking_visibility'] = st.toggle("إظهار عملية التفكير", value=prefs['thinking_visibility'])
        
        with st.expander("📜 تاريخ الحسابات"):
            if st.session_state.calculation_history:
                for i, calc in enumerate(reversed(st.session_state.calculation_history)):
                    st.json(calc, expanded=False)
            else:
                st.info("لا يوجد تاريخ حسابات بعد.")

if __name__ == "__main__":
    main()
