# ==============================================================================
#  العرّاب للجينات - الإصدار 15.0 (الوكيل المستقل)
#  - بنية هجينة ذكية تعتمد على قاعدة معرفة محلية موثوقة.
#  - مصمم ليعمل بكفاءة على بيئات التشغيل المحدودة.
# ==============================================================================

import streamlit as st
import collections
from itertools import product
from datetime import datetime
from typing import Dict, Tuple

# -------------------------------------------------
#  1. إعدادات الصفحة والتصميم
# -------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="العرّاب للجينات V15.0",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# --- CSS متقدم للواجهة العصرية ---
st.markdown("""
<style>
    .stDeployButton, #MainMenu, footer, header {visibility: hidden;}
    .block-container { padding: 0 !important; }
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .chat-container { background: rgba(255, 255, 255, 0.9); border-radius: 20px; padding: 0; margin: 20px auto; max-width: 1000px; height: 95vh; display: flex; flex-direction: column; box-shadow: 0 20px 40px rgba(0,0,0,0.1); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.2); }
    .header-bar { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 25px; display: flex; align-items: center; justify-content: space-between; border-radius: 20px 20px 0 0; flex-shrink: 0; }
    .header-title { font-size: 24px; font-weight: bold; margin: 0; display: flex; align-items: center; gap: 15px; }
    .status-indicator { width: 10px; height: 10px; background: #00ff88; border-radius: 50%; animation: pulse 2s infinite; box-shadow: 0 0 8px #00ff88; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    .chat-area { flex-grow: 1; overflow-y: auto; padding: 20px 30px; }
    .message { margin-bottom: 20px; animation: slideIn 0.3s ease-out; }
    @keyframes slideIn { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }
    .user-message { display: flex; justify-content: flex-end; }
    .user-bubble { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 18px; border-radius: 20px 20px 5px 20px; max-width: 80%; word-wrap: break-word; }
    .assistant-message { display: flex; align-items: flex-start; gap: 15px; }
    .avatar { width: 40px; height: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 20px; color: white; flex-shrink: 0; }
    .assistant-bubble { background: #f1f3f5; border: 1px solid #e9ecef; padding: 15px 20px; border-radius: 20px 20px 20px 5px; max-width: calc(100% - 55px); word-wrap: break-word; }
    .input-area { padding: 15px 25px; background: #ffffff; border-radius: 0 0 20px 20px; border-top: 1px solid #e9ecef; flex-shrink: 0; }
    .quick-actions { display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }
    .quick-btn { background: #e9ecef; border: none; color: #495057; padding: 6px 14px; border-radius: 15px; cursor: pointer; transition: all 0.2s ease; font-size: 13px; }
    .quick-btn:hover { background: #dee2e6; transform: translateY(-1px); }
    .genetics-calculator { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 15px; padding: 20px; margin-top: 15px; }
    .calc-header { font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #495057; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
#  2. قاعدة المعرفة الداخلية والوكيل الذكي
# -------------------------------------------------

GENE_DATA = {
    'B': {'display_name_ar': "اللون الأساسي", 'type_en': 'sex-linked', 'emoji': '🎨', 'alleles': {'BA': 'آش ريد', '+': 'أزرق/أسود', 'b': 'بني'}, 'dominance': ['BA', '+', 'b']},
    'd': {'display_name_ar': "التخفيف", 'type_en': 'sex-linked', 'emoji': '💧', 'alleles': {'+': 'عادي', 'd': 'مخفف'}, 'dominance': ['+', 'd']},
    'e': {'display_name_ar': "أحمر متنحي", 'type_en': 'autosomal', 'emoji': '🔴', 'alleles': {'+': 'عادي', 'e': 'أحمر متنحي'}, 'dominance': ['+', 'e']},
    'C': {'display_name_ar': "النمط", 'type_en': 'autosomal', 'emoji': '📐', 'alleles': {'CT': 'نمط تي', 'C': 'تشيكر', '+': 'بار', 'c': 'بدون بار'}, 'dominance': ['CT', 'C', '+', 'c']},
    'S': {'display_name_ar': "الانتشار", 'type_en': 'autosomal', 'emoji': '🌊', 'alleles': {'S': 'منتشر', '+': 'عادي'}, 'dominance': ['S', '+']}
}
GENE_ORDER = list(GENE_DATA.keys())
NAME_TO_SYMBOL_MAP = {g: {n: s for s, n in d['alleles'].items()} for g, d in GENE_DATA.items()}

class LocalAgent:
    def __init__(self, knowledge):
        self.knowledge = knowledge

    def understand_intent(self, query: str) -> Dict:
        query_lower = query.lower()
        if any(k in query_lower for k in ['احسب', 'حساب', 'نتائج', 'تزاوج']): return {'type': 'calculation'}
        if any(k in query_lower for k in ['مقارنة', 'فرق']): return {'type': 'comparison'}
        if any(k in query_lower for k in ['شرح', 'وضح', 'ما هو']): return {'type': 'explanation'}
        return {'type': 'general'}

    def generate_response(self, query: str) -> Dict:
        intent = self.understand_intent(query)
        if intent['type'] == 'calculation': return {"answer": "بالتأكيد! يسعدني أن أساعدك في حساب النتائج. تفضل باستخدام الحاسبة الوراثية المدمجة أدناه.", "show_calculator": True}
        
        mentioned_genes = [name for name, data in self.knowledge.items() if name.lower() in query.lower() or any(a.lower() in query.lower() for a in data['alleles'].values())]
        
        if not mentioned_genes:
            return {"answer": "🤔 لم أتمكن من تحديد الجين الذي تسأل عنه. هل يمكنك توضيح سؤالك؟ جرب استخدام أسماء الجينات مثل 'Spread' أو 'Checker'.", "show_calculator": False}

        response_parts = []
        for gene in mentioned_genes:
            info = self.knowledge[gene]
            part = f"🧬 **معلومات عن جين {info['display_name_ar']} ({gene})**:\n\n"
            part += f"- **النوع:** {info['type_en']}\n"
            part += f"- **الأليلات المتاحة:** {', '.join(info['alleles'].values())}\n"
            part += f"- **ترتيب السيادة:** {' > '.join(info['dominance'])}\n\n"
            response_parts.append(part)

        return {"answer": "\n".join(response_parts), "show_calculator": False}

class GeneticCalculator:
    def describe_phenotype(self, gt_dict: Dict) -> Tuple[str, str]:
        phenotypes = {g: "" for g in GENE_ORDER}
        for gene, gt_part in gt_dict.items():
            alleles = gt_part.replace('•//', '').split('//')
            for dom_allele in GENE_DATA[gene]['dominance']:
                if dom_allele in alleles: phenotypes[gene] = GENE_DATA[gene]['alleles'][dom_allele]; break
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
                    info, vis = GENE_DATA[gene], parent_inputs[parent].get(f'{gene}_visible')
                    hid = parent_inputs[parent].get(f'{gene}_hidden', vis)
                    vis_sym, hid_sym = NAME_TO_SYMBOL_MAP[gene].get(vis, info['dominance'][0]), NAME_TO_SYMBOL_MAP[gene].get(hid, vis)
                    if info['type_en'] == 'sex-linked' and parent == 'female': gt_parts.append(f"•//{vis_sym}")
                    else: gt_parts.append(f"{sorted([vis_sym, hid_sym], key=lambda x: info['dominance'].index(x))[0]}//{sorted([vis_sym, hid_sym], key=lambda x: info['dominance'].index(x))[1]}")
                parent_gts[parent] = gt_parts
            def get_gametes(gt_parts, is_female):
                parts = [[p.replace('•//','').strip()] if GENE_DATA[GENE_ORDER[i]]['type_en'] == 'sex-linked' and is_female else p.split('//') for i, p in enumerate(gt_parts)]
                return list(product(*parts))
            male_gametes, female_gametes = get_gametes(parent_gts['male'], False), get_gametes(parent_gts['female'], True)
            offspring = collections.Counter()
            for m_g in male_gametes:
                for f_g in female_gametes:
                    son, daughter = {}, {}
                    for i, gene in enumerate(GENE_ORDER):
                        alleles = sorted([m_g[i], f_g[i]], key=lambda x: GENE_DATA[gene]['dominance'].index(x))
                        if GENE_DATA[gene]['type_en'] == 'sex-linked': son[gene], daughter[gene] = f"{alleles[0]}//{alleles[1]}", f"•//{m_g[i]}"
                        else: son[gene], daughter[gene] = f"{alleles[0]}//{alleles[1]}", f"{alleles[0]}//{alleles[1]}"
                    offspring[self.describe_phenotype(son)] += 1
                    offspring[self.describe_phenotype(daughter)] += 1
            return {'results': offspring, 'total': sum(offspring.values())}
        except Exception as e: return {'error': f"خطأ في الحساب: {e}"}

# -------------------------------------------------
#  3. الواجهة الرئيسية
# -------------------------------------------------
def main():
    if "messages" not in st.session_state: st.session_state.messages = []
    
    agent = LocalAgent(GENE_DATA)

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown('<div class="header-bar"><div class="header-title">🧬 العرّاب للجينات V15.0</div><div style="font-size: 14px; display: flex; align-items: center; gap: 8px;"><div class="status-indicator"></div>نشط الآن</div></div>', unsafe_allow_html=True)

    chat_area = st.container()
    with chat_area:
        st.markdown('<div class="chat-area">', unsafe_allow_html=True)
        if not st.session_state.messages: st.session_state.messages.append({"role": "assistant", "content": "مرحباً بك في العرّاب V15.0! أنا وكيلك الذكي المستقل. كيف يمكنني مساعدتك اليوم؟"})
        for msg in st.session_state.messages:
            role_class = "user-message" if msg["role"] == "user" else "assistant-message"
            bubble_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
            avatar = "" if msg["role"] == "user" else '<div class="avatar">🤖</div>'
            st.markdown(f'<div class="message {role_class}">{avatar}<div class="{bubble_class}">{msg["content"]}</div></div>', unsafe_allow_html=True)
            if msg.get("show_calculator"): render_embedded_calculator()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    quick_actions = ["🧮 حساب وراثي", "🎨 شرح لون Spread", "💡 نصائح تربية"]
    cols = st.columns(len(quick_actions))
    for i, action in enumerate(cols):
        if action.button(quick_actions[i], use_container_width=True, key=f"quick_{i}"):
            handle_user_message(quick_actions[i], agent)
            st.rerun()
    if prompt := st.chat_input("اكتب سؤالك هنا... 💬"):
        handle_user_message(prompt, agent)
        st.rerun()
    st.markdown('</div></div>', unsafe_allow_html=True)

def handle_user_message(prompt, agent):
    st.session_state.messages.append({"role": "user", "content": prompt})
    response_data = agent.generate_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response_data["answer"], "show_calculator": response_data.get("show_calculator", False)})

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
                    else: parent_inputs[parent][f'{gene}_hidden'] = parent_inputs[parent][f'{gene}_visible']
        if st.button("🚀 احسب النتائج", use_container_width=True, type="primary"):
            calculator = GeneticCalculator()
            result_data = calculator.calculate(parent_inputs)
            if 'error' in result_data: st.error(result_data['error'])
            else:
                df = pd.DataFrame([{'النمط الظاهري': p, 'النمط الوراثي': g, 'النسبة %': f"{(c/result_data['total'])*100:.1f}%"} for (p, g), c in result_data['results'].items()])
                st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
