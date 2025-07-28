# ==============================================================================
#  العرّاب للجينات - الإصدار 14.0 (المحرك المتكامل)
#  - يدمج الواجهة العصرية مع العقل متعدد النماذج والحاسبة المدمجة
# ==============================================================================

import streamlit as st
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
import gdown
import PyPDF2
import os
import tempfile
import requests
import json
import numpy as np
from typing import List, Dict, Tuple
import time
import hashlib
from datetime import datetime
from itertools import product
import collections

# --- التحقق من توفر المكتبات الاختيارية ---
try:
    from sentence_transformers import SentenceTransformer
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False

# -------------------------------------------------
#  1. إعدادات الصفحة والتصميم
# -------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="العرّاب للجينات V14.0",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# --- CSS متقدم للواجهة العصرية ---
st.markdown("""
<style>
    /* إخفاء العناصر الافتراضية */
    .stDeployButton, #MainMenu, footer, header {visibility: hidden;}
    .block-container { padding: 0 !important; }
    
    /* الخلفية والتخطيط العام */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* حاوية المحادثة الرئيسية */
    .chat-container {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 0;
        margin: 20px auto;
        max-width: 1000px;
        height: 95vh;
        display: flex;
        flex-direction: column;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* شريط العنوان */
    .header-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 25px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-radius: 20px 20px 0 0;
        flex-shrink: 0;
    }
    
    .header-title { font-size: 24px; font-weight: bold; margin: 0; display: flex; align-items: center; gap: 15px; }
    .status-indicator { width: 10px; height: 10px; background: #00ff88; border-radius: 50%; animation: pulse 2s infinite; box-shadow: 0 0 8px #00ff88; }
    
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    
    /* منطقة المحادثة */
    .chat-area { flex-grow: 1; overflow-y: auto; padding: 20px 30px; }
    
    /* رسائل المحادثة */
    .message { margin-bottom: 20px; animation: slideIn 0.3s ease-out; }
    @keyframes slideIn { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }
    
    .user-message { display: flex; justify-content: flex-end; }
    .user-bubble { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 18px; border-radius: 20px 20px 5px 20px; max-width: 80%; word-wrap: break-word; }
    
    .assistant-message { display: flex; align-items: flex-start; gap: 15px; }
    .avatar { width: 40px; height: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 20px; color: white; flex-shrink: 0; }
    .assistant-bubble { background: #f1f3f5; border: 1px solid #e9ecef; padding: 15px 20px; border-radius: 20px 20px 20px 5px; max-width: calc(100% - 55px); word-wrap: break-word; }
    
    /* منطقة الإدخال */
    .input-area { padding: 15px 25px; background: #ffffff; border-radius: 0 0 20px 20px; border-top: 1px solid #e9ecef; flex-shrink: 0; }
    
    /* أزرار التشغيل السريع */
    .quick-actions { display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }
    .quick-btn { background: #e9ecef; border: none; color: #495057; padding: 6px 14px; border-radius: 15px; cursor: pointer; transition: all 0.2s ease; font-size: 13px; }
    .quick-btn:hover { background: #dee2e6; transform: translateY(-1px); }
    
    /* الحاسبة الوراثية المدمجة */
    .genetics-calculator { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 15px; padding: 20px; margin-top: 15px; }
    .calc-header { font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #495057; }
    
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
#  2. قواعد البيانات والمدراء
# -------------------------------------------------

# --- قاعدة بيانات الجينات (للحاسبة) ---
GENE_DATA = {
    'B': {'display_name_ar': "اللون الأساسي", 'type_en': 'sex-linked', 'emoji': '🎨', 'alleles': {'BA': 'آش ريد', '+': 'أزرق/أسود', 'b': 'بني'}, 'dominance': ['BA', '+', 'b']},
    'd': {'display_name_ar': "التخفيف", 'type_en': 'sex-linked', 'emoji': '💧', 'alleles': {'+': 'عادي', 'd': 'مخفف'}, 'dominance': ['+', 'd']},
    'e': {'display_name_ar': "أحمر متنحي", 'type_en': 'autosomal', 'emoji': '🔴', 'alleles': {'+': 'عادي', 'e': 'أحمر متنحي'}, 'dominance': ['+', 'e']},
    'C': {'display_name_ar': "النمط", 'type_en': 'autosomal', 'emoji': '📐', 'alleles': {'CT': 'نمط تي', 'C': 'تشيكر', '+': 'بار', 'c': 'بدون بار'}, 'dominance': ['CT', 'C', '+', 'c']},
    'S': {'display_name_ar': "الانتشار", 'type_en': 'autosomal', 'emoji': '🌊', 'alleles': {'S': 'منتشر', '+': 'عادي'}, 'dominance': ['S', '+']}
}
GENE_ORDER = list(GENE_DATA.keys())
NAME_TO_SYMBOL_MAP = {g: {n: s for s, n in d['alleles'].items()} for g, d in GENE_DATA.items()}

# --- مدير نماذج الذكاء الاصطناعي ---
class AIModelManager:
    def __init__(self):
        self.models = {
            "gemini": {"name": "Google Gemini", "available": self._check_secret("GEMINI_API_KEY"), "priority": 1},
            "deepseek": {"name": "DeepSeek", "available": self._check_secret("DEEPSEEK_API_KEY"), "priority": 2},
        }
    def _check_secret(self, key: str) -> bool:
        try: return st.secrets.get(key) is not None and st.secrets[key] != ""
        except Exception: return False
    def get_available_models(self) -> List[str]:
        available = [model for model, config in self.models.items() if config["available"]]
        return sorted(available, key=lambda x: self.models[x]["priority"])

# --- الحاسبة الوراثية ---
class AdvancedGeneticCalculator:
    def describe_phenotype(self, gt_dict: Dict) -> Tuple[str, str]:
        phenotypes = {g: "" for g in GENE_ORDER}
        for gene, gt_part in gt_dict.items():
            alleles = gt_part.replace('•//', '').split('//')
            for dom_allele in GENE_DATA[gene]['dominance']:
                if dom_allele in alleles:
                    phenotypes[gene] = GENE_DATA[gene]['alleles'][dom_allele]
                    break
        if 'e//e' in gt_dict.get('e', ''): phenotypes['B'] = 'أحمر متنحي'; phenotypes['C'] = ''
        if 'S' in gt_dict.get('S', ''):
            if 'e//e' not in gt_dict.get('e', ''): phenotypes['C'] = 'منتشر'
        sex = "أنثى" if any('•' in gt_dict.get(g, '') for g, d in GENE_DATA.items() if d['type_en'] == 'sex-linked') else "ذكر"
        desc_parts = [phenotypes.get('B'), 'مخفف' if phenotypes.get('d') == 'مخفف' else None, phenotypes.get('C')]
        gt_str = " | ".join([gt_dict[g].strip() for g in GENE_ORDER])
        return f"{sex} {' '.join(filter(None, desc_parts))}", gt_str

    def calculate(self, parent_inputs: Dict) -> Dict:
        try:
            parent_gts = {}
            for parent in ['male', 'female']:
                gt_parts = []
                for gene in GENE_ORDER:
                    info = GENE_DATA[gene]
                    vis = parent_inputs[parent].get(f'{gene}_visible')
                    hid = parent_inputs[parent].get(f'{gene}_hidden', vis)
                    vis_sym = NAME_TO_SYMBOL_MAP[gene].get(vis, info['dominance'][0])
                    hid_sym = NAME_TO_SYMBOL_MAP[gene].get(hid, vis_sym)
                    if info['type_en'] == 'sex-linked' and parent == 'female':
                        gt_parts.append(f"•//{vis_sym}")
                    else:
                        alleles = sorted([vis_sym, hid_sym], key=lambda x: info['dominance'].index(x))
                        gt_parts.append(f"{alleles[0]}//{alleles[1]}")
                parent_gts[parent] = gt_parts
            
            def get_gametes(gt_parts, is_female):
                parts_for_prod = []
                for i, part in enumerate(gt_parts):
                    gene = GENE_ORDER[i]
                    if GENE_DATA[gene]['type_en'] == 'sex-linked' and is_female:
                        parts_for_prod.append([part.replace('•//','').strip()])
                    else:
                        parts_for_prod.append(part.split('//'))
                return list(product(*parts_for_prod))

            male_gametes = get_gametes(parent_gts['male'], False)
            female_gametes = get_gametes(parent_gts['female'], True)
            
            offspring = collections.Counter()
            for m_g in male_gametes:
                for f_g in female_gametes:
                    son, daughter = {}, {}
                    for i, gene in enumerate(GENE_ORDER):
                        alleles = sorted([m_g[i], f_g[i]], key=lambda x: GENE_DATA[gene]['dominance'].index(x))
                        if GENE_DATA[gene]['type_en'] == 'sex-linked':
                            son[gene] = f"{alleles[0]}//{alleles[1]}"
                            daughter[gene] = f"•//{m_g[i]}"
                        else:
                            gt = f"{alleles[0]}//{alleles[1]}"
                            son[gene], daughter[gene] = gt, gt
                    offspring[self.describe_phenotype(son)] += 1
                    offspring[self.describe_phenotype(daughter)] += 1
            
            total = sum(offspring.values())
            return {'results': offspring, 'total': total, 'parent_genotypes': parent_gts}
        except Exception as e:
            return {'error': f"خطأ في الحساب: {e}"}

# -------------------------------------------------
#  3. تحميل الموارد وبناء قاعدة المعرفة
# -------------------------------------------------
@st.cache_resource
def load_resources():
    resources = {"embedder": None, "knowledge_base": None}
    if VECTOR_SEARCH_AVAILABLE:
        resources["embedder"] = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    return resources

@st.cache_data(ttl=86400)
def build_knowledge_base(_embedder):
    if not _embedder: return None
    db_path = os.path.join(tempfile.gettempdir(), "text_knowledge_v14.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS knowledge (source TEXT, content TEXT UNIQUE)")
    if cursor.fetchone()[0] == 0:
        with st.spinner("تحديث قاعدة المعرفة..."):
            # ... (Logic to load from BOOK_LINKS) ...
            pass # Simplified for brevity
    cursor.execute("SELECT source, content FROM knowledge")
    all_docs = [{"source": r[0], "content": r[1]} for r in cursor.fetchall()]
    conn.close()
    if not all_docs: return None
    contents = [doc['content'] for doc in all_docs]
    embeddings = _embedder.encode(contents, show_progress_bar=False)
    return {"documents": all_docs, "embeddings": embeddings}

# -------------------------------------------------
#  4. نظام الرد الذكي
# -------------------------------------------------
class IntelligentResponder:
    def __init__(self, ai_manager, resources):
        self.ai_manager = ai_manager
        self.resources = resources
        self.available_models = ai_manager.get_available_models()

    def search_knowledge(self, query):
        if not self.resources.get("knowledge_base") or not self.resources.get("embedder"): return []
        kb = self.resources["knowledge_base"]
        query_embedding = self.resources["embedder"].encode([query])
        similarities = cosine_similarity(query_embedding, kb['embeddings'])[0]
        top_indices = np.argsort(similarities)[-3:][::-1]
        return [kb['documents'][i] for i in top_indices if similarities[i] > 0.4]

    def generate_response(self, query: str) -> Dict:
        intent = self.understand_intent(query)
        if intent['type'] == 'calculation':
            return {"answer": "بالتأكيد! تفضل باستخدام الحاسبة الوراثية أدناه لحساب النتائج.", "show_calculator": True, "sources": []}

        context_docs = self.search_knowledge(query)
        
        for model_key in self.available_models:
            answer = ""
            if model_key == "gemini":
                answer = self.get_gemini_response(query, context_docs)
            elif model_key == "deepseek":
                answer = self.get_deepseek_response(query)
            
            if "خطأ" not in answer:
                return {"answer": answer, "show_calculator": False, "sources": context_docs}
        
        return {"answer": "عذراً، جميع النماذج الذكية تواجه مشكلة حالياً. يرجى المحاولة لاحقاً.", "show_calculator": False, "sources": []}

    def understand_intent(self, query):
        if any(k in query.lower() for k in ['احسب', 'حساب', 'نتائج', 'تزاوج']):
            return {'type': 'calculation'}
        return {'type': 'general'}

    def get_gemini_response(self, query, context_docs):
        try:
            API_KEY = st.secrets["GEMINI_API_KEY"]
            API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
            context = "\n\n".join([f"Source: {doc['source']}\nContent: {doc['content']}" for doc in context_docs])
            prompt = f"Based ONLY on the context below, answer the user's question in Arabic.\n\nContext:\n{context}\n\nUser Question: {query}\n\nAnswer (in Arabic):"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            response = requests.post(API_URL, json=payload, timeout=20)
            response.raise_for_status()
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            return f"خطأ في Gemini: {e}"

    def get_deepseek_response(self, query):
        # ... (DeepSeek logic) ...
        return "خطأ في DeepSeek"

# -------------------------------------------------
#  5. الواجهة الرئيسية
# -------------------------------------------------
def main():
    initialize_session_state()
    resources = load_resources()
    ai_manager = AIModelManager()
    responder = IntelligentResponder(ai_manager, resources)

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown(f'''
    <div class="header-bar">
        <div class="header-title">🧬 العرّاب للجينات V14.0</div>
        <div style="font-size: 14px; display: flex; align-items: center; gap: 8px;">
            <div class="status-indicator"></div>
            نشط الآن
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Chat Area
    chat_area = st.container()
    with chat_area:
        st.markdown('<div class="chat-area">', unsafe_allow_html=True)
        if not st.session_state.messages:
            st.session_state.messages.append({"role": "assistant", "content": "مرحباً بك في العرّاب V14.0! كيف يمكنني مساعدتك في عالم وراثة الحمام اليوم؟"})
        
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="message user-message"><div class="user-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="message assistant-message"><div class="avatar">🤖</div><div class="assistant-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
                if msg.get("show_calculator"):
                    render_embedded_calculator()
        st.markdown('</div>', unsafe_allow_html=True)

    # Input Area
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    quick_actions = ["🧮 حساب وراثي", "🎨 شرح الألوان", "💡 نصائح تربية"]
    cols = st.columns(len(quick_actions))
    for i, action in enumerate(cols):
        if action.button(quick_actions[i], use_container_width=True):
            handle_user_message(quick_actions[i], responder)
            st.rerun()

    if prompt := st.chat_input("اكتب سؤالك هنا... 💬"):
        handle_user_message(prompt, responder)
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def handle_user_message(prompt, responder):
    st.session_state.messages.append({"role": "user", "content": prompt})
    response_data = responder.generate_response(prompt)
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_data["answer"],
        "show_calculator": response_data.get("show_calculator", False),
        "sources": response_data.get("sources", [])
    })

def render_embedded_calculator():
    with st.container():
        st.markdown('<div class="genetics-calculator">', unsafe_allow_html=True)
        st.markdown('<div class="calc-header">🧮 الحاسبة الوراثية المدمجة</div>', unsafe_allow_html=True)
        
        parent_inputs = {'male': {}, 'female': {}}
        col1, col2 = st.columns(2)
        
        for parent, col in [('male', col1), ('female', col2)]:
            with col:
                st.markdown(f"#### {'♂️ الذكر' if parent == 'male' else '♀️ الأنثى'}")
                for gene, data in GENE_DATA.items():
                    choices = list(data['alleles'].values())
                    parent_inputs[parent][f'{gene}_visible'] = st.selectbox(f"{data['emoji']} {data['display_name_ar']} (الظاهر):", choices, key=f"emb_{parent}_{gene}_vis")
                    if not (data['type_en'] == 'sex-linked' and parent == 'female'):
                        parent_inputs[parent][f'{gene}_hidden'] = st.selectbox(f"{data['emoji']} {data['display_name_ar']} (المخفي):", choices, key=f"emb_{parent}_{gene}_hid", index=choices.index(parent_inputs[parent][f'{gene}_visible']))
                    else:
                        parent_inputs[parent][f'{gene}_hidden'] = parent_inputs[parent][f'{gene}_visible']
        
        if st.button("🚀 احسب النتائج", use_container_width=True, type="primary"):
            calculator = AdvancedGeneticCalculator()
            result_data = calculator.calculate(parent_inputs)
            if 'error' in result_data:
                st.error(result_data['error'])
            else:
                df = pd.DataFrame([{'النمط الظاهري': p, 'النمط الوراثي': g, 'النسبة %': f"{(c/result_data['total'])*100:.1f}%"} for (p, g), c in result_data['results'].items()])
                st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

if __name__ == "__main__":
    main()
