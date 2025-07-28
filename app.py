# ==============================================================================
#  العرّاب للجينات - الإصدار 16.0 (مع ذاكرة Google Drive)
#  - يحفظ ويسترجع سجل المحادثات من Google Drive لضمان الاستمرارية.
# ==============================================================================

import streamlit as st
import collections
from itertools import product
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import json
import os
import tempfile

# --- التحقق من توفر مكتبات Google ---
try:
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
    import io
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

# -------------------------------------------------
#  1. إعدادات الصفحة والتصميم
# -------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="العرّاب للجينات V16.0",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# --- CSS متقدم للواجهة العصرية ---
st.markdown("""
<style>
    /* ... (نفس كود CSS من الإصدار 15.0) ... */
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
#  2. مدير Google Drive
# -------------------------------------------------

@st.cache_resource
class GoogleDriveManager:
    def __init__(self):
        self.creds = self._get_credentials()
        self.drive_service = self._build_service()
        self.file_id = None

    def _get_credentials(self):
        if not GOOGLE_AVAILABLE: return None
        try:
            # قراءة بيانات الاعتماد من "خزنة الأسرار"
            creds_json = {
                "type": st.secrets["gcp_service_account"]["type"],
                "project_id": st.secrets["gcp_service_account"]["project_id"],
                "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
                "private_key": st.secrets["gcp_service_account"]["private_key"],
                "client_email": st.secrets["gcp_service_account"]["client_email"],
                "client_id": st.secrets["gcp_service_account"]["client_id"],
                "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
                "token_uri": st.secrets["gcp_service_account"]["token_uri"],
                "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
                "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"]
            }
            scopes = ['https://www.googleapis.com/auth/drive']
            return Credentials.from_service_account_info(creds_json, scopes=scopes)
        except Exception:
            return None

    def _build_service(self):
        if self.creds:
            return build('drive', 'v3', credentials=self.creds)
        return None

    def is_connected(self):
        return self.drive_service is not None

    def _find_file(self, filename="alarrab_chat_history.json"):
        if not self.is_connected(): return None
        try:
            response = self.drive_service.files().list(
                q=f"name='{filename}' and trashed=false",
                spaces='drive',
                fields='files(id, name)').execute()
            files = response.get('files', [])
            return files[0]['id'] if files else None
        except Exception:
            return None

    def load_chat_history(self, filename="alarrab_chat_history.json"):
        if not self.is_connected(): return []
        self.file_id = self._find_file(filename)
        if not self.file_id: return []
        
        try:
            request = self.drive_service.files().get_media(fileId=self.file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            fh.seek(0)
            return json.load(fh)
        except Exception:
            return []

    def save_chat_history(self, messages, filename="alarrab_chat_history.json"):
        if not self.is_connected(): return
        
        try:
            file_metadata = {'name': filename}
            media_body = MediaFileUpload(self._create_temp_json_file(messages), mimetype='application/json')
            
            if self.file_id:
                self.drive_service.files().update(fileId=self.file_id, media_body=media_body).execute()
            else:
                file = self.drive_service.files().create(body=file_metadata, media_body=media_body, fields='id').execute()
                self.file_id = file.get('id')
        except Exception as e:
            st.error(f"فشل حفظ المحادثة في Google Drive: {e}")

    def _create_temp_json_file(self, messages):
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, "temp_history.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
        return file_path

# -------------------------------------------------
#  3. الوكيل الذكي وقاعدة المعرفة
# -------------------------------------------------
# --- (نفس كود الوكيل الذكي والحاسبة من الإصدار 15.0) ---
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
    def __init__(self, knowledge): self.knowledge = knowledge
    def understand_intent(self, query: str) -> Dict:
        query_lower = query.lower()
        if any(k in query_lower for k in ['احسب', 'حساب', 'نتائج', 'تزاوج']): return {'type': 'calculation'}
        if any(k in query_lower for k in ['مقارنة', 'فرق']): return {'type': 'comparison'}
        if any(k in query_lower for k in ['شرح', 'وضح', 'ما هو']): return {'type': 'explanation'}
        return {'type': 'general'}
    def generate_response(self, query: str) -> Dict:
        intent = self.understand_intent(query)
        if intent['type'] == 'calculation': return {"answer": "بالتأكيد! تفضل باستخدام الحاسبة الوراثية المدمجة أدناه.", "show_calculator": True}
        mentioned_genes = [name for name, data in self.knowledge.items() if name.lower() in query.lower() or any(a.lower() in query.lower() for a in data['alleles'].values())]
        if not mentioned_genes: return {"answer": "🤔 لم أتمكن من تحديد الجين الذي تسأل عنه. جرب استخدام أسماء الجينات مثل 'Spread' أو 'Checker'.", "show_calculator": False}
        response_parts = []
        for gene in mentioned_genes:
            info = self.knowledge[gene]
            part = f"🧬 **معلومات عن جين {info['display_name_ar']} ({gene})**:\n\n- **النوع:** {info['type_en']}\n- **الأليلات:** {', '.join(info['alleles'].values())}\n- **السيادة:** {' > '.join(info['dominance'])}"
            response_parts.append(part)
        return {"answer": "\n\n".join(response_parts), "show_calculator": False}
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
#  4. الواجهة الرئيسية
# -------------------------------------------------
def main():
    drive_manager = GoogleDriveManager()
    agent = LocalAgent(GENE_DATA)

    # تحميل المحادثة عند بدء الجلسة
    if "messages" not in st.session_state:
        if drive_manager.is_connected():
            st.session_state.messages = drive_manager.load_chat_history()
        else:
            st.session_state.messages = []

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    header_status = "متصل بالذاكرة السحابية" if drive_manager.is_connected() else "الوضع المحلي"
    st.markdown(f'<div class="header-bar"><div class="header-title">🧬 العرّاب للجينات V16.0</div><div style="font-size: 14px; display: flex; align-items: center; gap: 8px;"><div class="status-indicator"></div>{header_status}</div></div>', unsafe_allow_html=True)

    chat_area = st.container()
    with chat_area:
        st.markdown('<div class="chat-area">', unsafe_allow_html=True)
        if not st.session_state.messages:
            st.session_state.messages.append({"role": "assistant", "content": "مرحباً بك في العرّاب V16.0! أنا وكيلك الذكي ذو الذاكرة الدائمة. كيف يمكنني مساعدتك اليوم؟"})
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
            handle_user_message(quick_actions[i], agent, drive_manager)
            st.rerun()
    if prompt := st.chat_input("اكتب سؤالك هنا... 💬"):
        handle_user_message(prompt, agent, drive_manager)
        st.rerun()
    st.markdown('</div></div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("☁️ الذاكرة السحابية")
        if drive_manager.is_connected():
            st.success("متصل بـ Google Drive")
            st.info("يتم حفظ سجل المحادثة تلقائيًا.")
        else:
            st.warning("غير متصل بـ Google Drive")
            st.caption("لن يتم حفظ المحادثة بين الجلسات. اتبع دليل الإعداد للتفعيل.")

def handle_user_message(prompt, agent, drive_manager):
    st.session_state.messages.append({"role": "user", "content": prompt})
    response_data = agent.generate_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response_data["answer"], "show_calculator": response_data.get("show_calculator", False)})
    # حفظ المحادثة بعد كل رسالة
    drive_manager.save_chat_history(st.session_state.messages)

def render_embedded_calculator():
    # ... (نفس كود الحاسبة من الإصدار 15.0) ...
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
