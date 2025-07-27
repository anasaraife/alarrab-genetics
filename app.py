# ===================================================================
# 🧬 العرّاب للجينات V7.2 - استعادة الواجهة الاحترافية
# تم إعادة دمج الواجهة العصرية التي صممها المستخدم مع إصلاح الأخطاء.
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
from typing import List, Dict, Tuple
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

# --- إعدادات الصفحة ---
st.set_page_config(
    layout="wide",
    page_title="العرّاب للجينات V7.2",
    page_icon="🧬",
    initial_sidebar_state="auto"
)

# --- CSS متقدم للواجهة العصرية ---
st.markdown("""
<style>
    /* إخفاء العناصر الافتراضية */
    .stDeployButton, #MainMenu, footer, header {visibility: hidden;}
    
    /* الخلفية والتخطيط العام */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* حاوية المحادثة الرئيسية */
    .chat-container {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        padding: 0;
        margin: 10px auto;
        max-width: 1200px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.08);
        overflow: hidden;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* شريط العنوان */
    .header-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px 30px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-radius: 20px 20px 0 0;
    }
    
    .header-title {
        font-size: 28px;
        font-weight: bold;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .status-indicator {
        width: 12px;
        height: 12px;
        background: #00ff88;
        border-radius: 50%;
        animation: pulse 2s infinite;
        box-shadow: 0 0 10px #00ff88;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* منطقة المحادثة */
    .chat-area {
        height: 75vh;
        overflow-y: auto;
        padding: 20px 30px;
        background: #ffffff;
    }
    
    /* رسائل المحادثة */
    .message {
        margin-bottom: 25px;
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin-left: 80px;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 25px 25px 5px 25px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        max-width: 100%;
        word-wrap: break-word;
    }
    
    .assistant-message {
        display: flex;
        align-items: flex-start;
        gap: 15px;
        margin-right: 80px;
    }
    
    .avatar {
        width: 45px;
        height: 45px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        color: white;
        box-shadow: 0 4px 15px rgba(118, 75, 162, 0.3);
        flex-shrink: 0;
    }
    
    .assistant-bubble {
        background: #f1f3f5;
        border: 1px solid #e9ecef;
        padding: 20px;
        border-radius: 25px 25px 25px 5px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        max-width: 100%;
        word-wrap: break-word;
        position: relative;
    }
    
    /* منطقة الإدخال */
    .input-area {
        padding: 20px 30px;
        background: #f8f9fa;
        border-radius: 0 0 20px 20px;
        border-top: 1px solid #e9ecef;
    }
    
    /* مؤشر الكتابة */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 15px 20px;
        background: #f1f3f5;
        border-radius: 25px 25px 25px 5px;
        margin-right: 80px;
        margin-left: 60px;
    }
    
    .typing-dots {
        display: flex;
        gap: 4px;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background: #667eea;
        border-radius: 50%;
        animation: typingBounce 1.4s infinite ease-in-out;
    }
    
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typingBounce {
        0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
        40% { transform: scale(1); opacity: 1; }
    }
    
    /* الحاسبة الوراثية المدمجة */
    .genetics-calculator {
        background: #ffffff;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        color: #333;
        border: 1px solid #dee2e6;
    }
    
    .calc-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 20px;
        font-size: 18px;
        font-weight: bold;
        color: #4a4a4a;
    }
    
    .result-card {
        background: #f8f9fa;
        color: #333;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    .message-timestamp {
        font-size: 12px;
        color: #999;
        margin-top: 8px;
        text-align: right;
    }
    .assistant-message .message-timestamp {
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

# --- قواعد البيانات المحسنة ---
GENE_DATA = {
    'B': {
        'display_name_ar': "اللون الأساسي", 'type_en': 'sex-linked', 'emoji': '🎨',
        'alleles': {
            'BA': {'name': 'آش ريد'},
            '+': {'name': 'أزرق/أسود'},
            'b': {'name': 'بني'}
        },
        'dominance': ['BA', '+', 'b']
    },
    'd': {
        'display_name_ar': "التخفيف", 'type_en': 'sex-linked', 'emoji': '💧',
        'alleles': {
            '+': {'name': 'عادي (غير مخفف)'},
            'd': {'name': 'مخفف'}
        },
        'dominance': ['+', 'd']
    },
    'e': {
        'display_name_ar': "أحمر متنحي", 'type_en': 'autosomal', 'emoji': '🔴',
        'alleles': {
            '+': {'name': 'عادي (غير أحمر متنحي)'},
            'e': {'name': 'أحمر متنحي'}
        },
        'dominance': ['+', 'e']
    },
    'C': {
        'display_name_ar': "النمط", 'type_en': 'autosomal', 'emoji': '📐',
        'alleles': {
            'CT': {'name': 'نمط تي (مخملي)'},
            'C': {'name': 'تشيكر'},
            '+': {'name': 'بار (شريط)'},
            'c': {'name': 'بدون شريط'}
        },
        'dominance': ['CT', 'C', '+', 'c']
    },
    'S': {
        'display_name_ar': "الانتشار (سبريد)", 'type_en': 'autosomal', 'emoji': '🌊',
        'alleles': {
            'S': {'name': 'منتشر (سبريد)'},
            '+': {'name': 'عادي (غير منتشر)'}
        },
        'dominance': ['S', '+']
    }
}
GENE_ORDER = list(GENE_DATA.keys())
NAME_TO_SYMBOL_MAP = {
    gene: {info['name']: symbol for symbol, info in data['alleles'].items()}
    for gene, data in GENE_DATA.items()
}

# --- إدارة الجلسة المحسنة ---
def initialize_session_state():
    """تهيئة متغيرات الجلسة."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "typing" not in st.session_state:
        st.session_state.typing = False

# --- تحميل الموارد المحسن ---
@st.cache_resource
def load_resources():
    """تحميل جميع الموارد اللازمة للوكيل الذكي."""
    resources = {"status": "limited"}
    
    if VECTOR_SEARCH_AVAILABLE:
        vector_db_path = "pigeon_knowledge_base_v8.0.pkl"
        if os.path.exists(vector_db_path):
            try:
                with open(vector_db_path, "rb") as f:
                    resources["vector_db"] = pickle.load(f)
                resources["embedder"] = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            except Exception as e:
                st.warning(f"⚠️ تعذر تحميل قاعدة المتجهات: {e}")
    
    if GEMINI_AVAILABLE and "GEMINI_API_KEY" in st.secrets:
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            resources["model"] = genai.GenerativeModel('gemini-1.5-flash',
                generation_config={"temperature": 0.1, "max_output_tokens": 3000})
            resources["status"] = "ready"
        except Exception as e:
            st.error(f"❌ فشل تفعيل الذكاء الاصطناعي: {e}")
            resources["status"] = "error"
            
    return resources

# --- فئة الحاسبة الوراثية المحسنة ---
class AdvancedGeneticCalculator:
    def describe_phenotype(self, genotype_dict: Dict[str, str]) -> Tuple[str, str]:
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

    def calculate_advanced_genetics(self, parent_inputs: Dict) -> Dict:
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
    def __init__(self, resources: Dict):
        self.resources = resources
        self.calculator = AdvancedGeneticCalculator()

    def understand_query(self, query: str) -> Dict:
        intent = {'type': 'general', 'calculation_needed': False}
        if any(keyword in query.lower() for keyword in ['احسب', 'حساب', 'نتائج', 'تزاوج', 'تربية']):
            intent['type'] = 'calculation'
            intent['calculation_needed'] = True
        elif any(keyword in query.lower() for keyword in ['شرح', 'وضح', 'كيف', 'ماذا', 'لماذا']):
            intent['type'] = 'explanation'
        elif any(keyword in query.lower() for keyword in ['مساعدة', 'help', 'مرحبا', 'السلام']):
            intent['type'] = 'greeting'
        return intent

    def search_deep_memory(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.resources.get("vector_db") or not self.resources.get("embedder"): return []
        try:
            index = self.resources["vector_db"]["index"]
            chunks = self.resources["vector_db"]["chunks"]
            query_embedding = self.resources["embedder"].encode([query])
            distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
            return [{"content": chunks[idx], "relevance": 1 / (1 + dist)} for dist, idx in zip(distances[0], indices[0]) if idx < len(chunks)]
        except Exception:
            return []

    def generate_smart_response(self, query: str, intent: Dict) -> Dict:
        if not self.resources.get("model"):
            return {"answer": "❌ عذراً، نظام الذكاء الاصطناعي غير متاح حالياً.", "calculation_widget": intent['calculation_needed']}
        
        deep_results = self.search_deep_memory(query)
        context = "\n\n".join([f"معلومة: {r['content']}" for r in deep_results[:3]])
        
        system_prompt = "أنت 'العرّاب للجينات V7.2'، وكيل ذكاء اصطناعي متخصص في وراثة الحمام..."
        user_prompt = f"سؤال: {query}\nالسياق: {context}"

        try:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.resources["model"].generate_content(full_prompt)
            return {"answer": response.text, "sources": deep_results, "calculation_widget": intent['calculation_needed']}
        except Exception as e:
            return {"answer": f"❌ عذراً، حدث خطأ: {str(e)}", "sources": deep_results, "calculation_widget": intent['calculation_needed']}

# --- الحاسبة المدمجة العصرية ---
def render_embedded_calculator():
    with st.expander("🧮 الحاسبة الوراثية المدمجة", expanded=True):
        st.markdown('<div class="genetics-calculator">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        parent_inputs = {'male': {}, 'female': {}}
        
        with col1:
            st.markdown("#### ♂️ **الذكر (الأب)**")
            for gene, data in GENE_DATA.items():
                with st.container():
                    st.write(f"**{data['emoji']} {data['display_name_ar']}**")
                    choices = ["(اختر الصفة)"] + [v['name'] for v in data['alleles'].values()]
                    parent_inputs['male'][f'{gene}_visible'] = st.selectbox("الصفة الظاهرة:", choices, key=f"emb_male_{gene}_visible", label_visibility="collapsed")
                    parent_inputs['male'][f'{gene}_hidden'] = st.selectbox("الصفة المخفية:", choices, key=f"emb_male_{gene}_hidden", label_visibility="collapsed")
        
        with col2:
            st.markdown("#### ♀️ **الأنثى (الأم)**")
            for gene, data in GENE_DATA.items():
                 with st.container():
                    st.write(f"**{data['emoji']} {data['display_name_ar']}**")
                    choices = ["(اختر الصفة)"] + [v['name'] for v in data['alleles'].values()]
                    parent_inputs['female'][f'{gene}_visible'] = st.selectbox("الصفة الظاهرة:", choices, key=f"emb_female_{gene}_visible", label_visibility="collapsed")
                    if data['type_en'] != 'sex-linked':
                        parent_inputs['female'][f'{gene}_hidden'] = st.selectbox("الصفة المخفية:", choices, key=f"emb_female_{gene}_hidden", label_visibility="collapsed")
                    else:
                        st.info("الإناث لديها أليل واحد فقط")
                        parent_inputs['female'][f'{gene}_hidden'] = parent_inputs['female'][f'{gene}_visible']
        
        if st.button("🚀 احسب النتائج المتوقعة", use_container_width=True, type="primary"):
            if not all([parent_inputs['male'].get('B_visible') != "(اختر الصفة)", parent_inputs['female'].get('B_visible') != "(اختر الصفة)"]):
                st.error("⚠️ يرجى اختيار اللون الأساسي لكلا الوالدين")
            else:
                with st.spinner("🧬 جاري الحساب..."):
                    calculator = AdvancedGeneticCalculator()
                    result_data = calculator.calculate_advanced_genetics(parent_inputs)
                    if 'error' in result_data:
                        st.error(result_data['error'])
                    else:
                        display_advanced_results(result_data)
        st.markdown('</div>', unsafe_allow_html=True)


def display_advanced_results(result_data: Dict):
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
    
    # الشريط الجانبي
    with st.sidebar:
        st.markdown("### 🔧 اختر الأداة")
        # قائمة اختيار نوع الحاسبة
        tool = st.radio(
            "ما الذي تريد حسابه؟",
            ["حاسبة الألوان", "التركيبات الجينية", "إرشادات عامة"]
        )
        # خيارات إضافية حسب الأداة
        if tool == "حاسبة الألوان":
            base_color = st.selectbox("اختر اللون الأساسي", ["Blue", "Ash-Red", "Brown"])
            st.write("سيتم استخدام اللون:", base_color)
        elif tool == "التركيبات الجينية":
            parent1 = st.text_input("ادخل جينات الأب")
            parent2 = st.text_input("ادخل جينات الأم")
            if st.button("احسب"):
                st.success(f"تم حساب التهجين بين {parent1} و {parent2}")

    # حاوية الواجهة الرئيسية
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # شريط العنوان
    st.markdown(f'''
    <div class="header-bar">
        <div class="header-title">
            🧬 العرّاب للجينات V7.2
            <div class="status-indicator" style="background: {'#00ff88' if agent.resources['status'] == 'ready' else '#ffc107'};"></div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # منطقة المحادثة
    chat_area = st.container()
    with chat_area:
        st.markdown('<div class="chat-area">', unsafe_allow_html=True)
        
        if not st.session_state.messages:
            welcome_message = "🧬 **مرحباً بك في العرّاب للجينات V7.2!** أنا وكيلك الذكي المتخصص. كيف يمكنني مساعدتك؟"
            st.session_state.messages.append({"role": "assistant", "content": welcome_message, "show_calculator": False, "timestamp": datetime.now()})

        for message in st.session_state.messages:
            ts = message.get("timestamp", datetime.now()).strftime("%H:%M")
            if message["role"] == "user":
                st.markdown(f'<div class="message user-message"><div class="user-bubble">{message["content"]}<div class="message-timestamp">{ts}</div></div></div>', unsafe_allow_html=True)
            else: # assistant
                st.markdown(f'<div class="message assistant-message"><div class="avatar">🤖</div><div class="assistant-bubble">{message["content"]}<div class="message-timestamp">{ts}</div></div></div>', unsafe_allow_html=True)
                if message.get("show_calculator"):
                    render_embedded_calculator()
        
        if st.session_state.get('typing'):
             st.markdown('<div class="message assistant-message"><div class="avatar">🤖</div><div class="typing-indicator"><div class="typing-dots"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div><span style="margin-left: 10px; color: #666;">العرّاب يفكر...</span></div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # منطقة الإدخال
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    
    user_input = st.chat_input("اكتب سؤالك هنا... 💬", key="main_input")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input, "timestamp": datetime.now()})
        st.session_state.typing = True
        st.rerun()
        
    st.markdown('</div></div>', unsafe_allow_html=True)

    # معالجة الرسالة الأخيرة
    if st.session_state.typing:
        last_message = st.session_state.messages[-1]["content"]
        
        intent = agent.understand_query(last_message)
        response_data = agent.generate_smart_response(last_message, intent)
        
        assistant_message = {
            "role": "assistant",
            "content": response_data["answer"],
            "sources": response_data.get("sources", []),
            "show_calculator": response_data.get("calculation_widget", False),
            "timestamp": datetime.now()
        }
        st.session_state.messages.append(assistant_message)
        st.session_state.typing = False
        st.rerun()

if __name__ == "__main__":
    main()
