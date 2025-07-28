# ===================================================================
# 🧬 العرّاب للجينات V6.0 - وكيل الذكاء الاصطناعي المتقدم
# واجهة عصرية تشبه ChatGPT مع تكامل ذكي للحاسبة الوراثية
# ===================================================================

import streamlit as st
from itertools import product
import collections
import pandas as pd
import numpy as np
import pickle
import os
import json
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional
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
    page_title="العرّاب للجينات V6.0",
    page_icon="🧬",
    initial_sidebar_state="collapsed"
)

# --- CSS متقدم للواجهة العصرية ---
st.markdown("""
<style>
    /* إخفاء العناصر الافتراضية */
    .stDeployButton, #MainMenu, footer, header {visibility: hidden;}
    
    /* الخلفية والتخطيط العام */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* حاوية المحادثة الرئيسية */
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 0;
        margin: 20px auto;
        max-width: 1200px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        overflow: hidden;
    }
    
    /* شريط العنوان */
    .header-bar {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
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
        height: 70vh;
        overflow-y: auto;
        padding: 20px 30px;
        background: white;
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
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        color: white;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
        flex-shrink: 0;
    }
    
    .assistant-bubble {
        background: #f8f9fa;
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
    
    /* أزرار التشغيل السريع */
    .quick-actions {
        display: flex;
        gap: 10px;
        margin-bottom: 15px;
        flex-wrap: wrap;
    }
    
    .quick-btn {
        background: white;
        border: 2px solid #4facfe;
        color: #4facfe;
        padding: 8px 16px;
        border-radius: 20px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 14px;
        font-weight: 500;
    }
    
    .quick-btn:hover {
        background: #4facfe;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
    }
    
    /* مؤشر الكتابة */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 15px 20px;
        background: #f8f9fa;
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
        background: #4facfe;
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        color: white;
    }
    
    .calc-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 20px;
        font-size: 18px;
        font-weight: bold;
    }
    
    .gene-selector {
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        backdrop-filter: blur(5px);
    }
    
    .result-card {
        background: rgba(255, 255, 255, 0.95);
        color: #333;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* رسائل النظام */
    .system-message {
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 15px;
        margin: 15px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* شريط التقدم */
    .progress-bar {
        width: 100%;
        height: 4px;
        background: #e9ecef;
        border-radius: 2px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 2px;
        animation: progressSlide 2s ease-in-out;
    }
    
    @keyframes progressSlide {
        from { width: 0%; }
        to { width: 100%; }
    }
    
    /* تحسينات متجاوبة */
    @media (max-width: 768px) {
        .chat-container { margin: 10px; }
        .header-bar { padding: 15px 20px; }
        .header-title { font-size: 22px; }
        .chat-area { height: 60vh; padding: 15px 20px; }
        .user-message { margin-left: 20px; }
        .assistant-message { margin-right: 20px; }
        .input-area { padding: 15px 20px; }
    }
</style>
""", unsafe_allow_html=True)

# --- قواعد البيانات المحسنة ---
GENE_DATA = {
    'B': {
        'display_name_ar': "اللون الأساسي",
        'display_name_en': "Base Color",
        'type_en': 'sex-linked',
        'emoji': '🎨',
        'alleles': {
            'BA': {'name': 'آش ريد', 'name_en': 'Ash Red', 'description': 'اللون الأحمر الرمادي'},
            '+': {'name': 'أزرق/أسود', 'name_en': 'Blue/Black', 'description': 'اللون الأزرق أو الأسود الطبيعي'},
            'b': {'name': 'بني', 'name_en': 'Brown', 'description': 'اللون البني المتنحي'}
        },
        'dominance': ['BA', '+', 'b']
    },
    'd': {
        'display_name_ar': "التخفيف",
        'display_name_en': "Dilution",
        'type_en': 'sex-linked',
        'emoji': '💧',
        'alleles': {
            '+': {'name': 'عادي (غير مخفف)', 'name_en': 'Normal', 'description': 'بدون تخفيف اللون'},
            'd': {'name': 'مخفف', 'name_en': 'Dilute', 'description': 'لون مخفف فاتح'}
        },
        'dominance': ['+', 'd']
    },
    'e': {
        'display_name_ar': "أحمر متنحي",
        'display_name_en': "Recessive Red",
        'type_en': 'autosomal',
        'emoji': '🔴',
        'alleles': {
            '+': {'name': 'عادي (غير أحمر متنحي)', 'name_en': 'Normal', 'description': 'يظهر الألوان الأخرى'},
            'e': {'name': 'أحمر متنحي', 'name_en': 'Recessive Red', 'description': 'يخفي جميع الألوان الأخرى'}
        },
        'dominance': ['+', 'e']
    },
    'C': {
        'display_name_ar': "النمط",
        'display_name_en': "Pattern",
        'type_en': 'autosomal',
        'emoji': '📐',
        'alleles': {
            'CT': {'name': 'نمط تي (مخملي)', 'name_en': 'T-Pattern', 'description': 'نمط الشطرنج المخملي'},
            'C': {'name': 'تشيكر', 'name_en': 'Checker', 'description': 'نمط الشطرنج'},
            '+': {'name': 'بار (شريط)', 'name_en': 'Bar', 'description': 'نمط الخطوط الطبيعي'},
            'c': {'name': 'بدون شريط', 'name_en': 'Barless', 'description': 'بدون أي خطوط'}
        },
        'dominance': ['CT', 'C', '+', 'c']
    },
    'S': {
        'display_name_ar': "الانتشار (سبريد)",
        'display_name_en': "Spread",
        'type_en': 'autosomal',
        'emoji': '🌊',
        'alleles': {
            'S': {'name': 'منتشر (سبريد)', 'name_en': 'Spread', 'description': 'لون منتشر بالكامل'},
            '+': {'name': 'عادي (غير منتشر)', 'name_en': 'Non-Spread', 'description': 'لون غير منتشر'}
        },
        'dominance': ['S', '+']
    }
}

GENE_ORDER = list(GENE_DATA.keys())

# خريطة تحويل الأسماء إلى الرموز
NAME_TO_SYMBOL_MAP = {
    gene: {info['name']: symbol for symbol, info in data['alleles'].items()}
    for gene, data in GENE_DATA.items()
}

# قوالب المحادثة الذكية
CONVERSATION_TEMPLATES = {
    'greeting': [
        "مرحباً! أنا العرّاب للجينات، خبيرك في وراثة الحمام. كيف يمكنني مساعدتك اليوم؟ 🧬",
        "أهلاً وسهلاً! استعد لاستكشاف عالم وراثة الحمام المثير معي! 🕊️",
        "مرحباً بك في عالم الجينات! أنا هنا للإجابة على جميع أسئلتك حول وراثة الحمام. 🎯"
    ],
    'calculation_request': [
        "دعني أساعدك في حساب النتائج الوراثية! سأحتاج لبعض المعلومات عن الوالدين...",
        "ممتاز! سأقوم بتشغيل الحاسبة الوراثية المتقدمة لك الآن...",
        "حسناً، دعنا نحسب التوقعات الوراثية بدقة علمية!"
    ],
    'explanation': [
        "هذا سؤال رائع! دعني أشرح لك بالتفصيل...",
        "مفهوم مهم جداً في علم الوراثة! إليك التفسير الكامل...",
        "سؤال علمي ممتاز! سأوضح لك كل شيء بطريقة مبسطة..."
    ]
}

# --- إدارة الجلسة المحسنة ---
def initialize_session_state():
    """تهيئة متغيرات الجلسة مع إعدادات متقدمة."""
    defaults = {
        "messages": [],
        "conversation_context": [],
        "current_calculation": None,
        "user_preferences": {
            "language": "ar",
            "detail_level": "medium",
            "show_genetics_formulas": True
        },
        "session_stats": {
            "queries": 0,
            "calculations": 0,
            "deep_searches": 0,
            "live_searches": 0,
            "start_time": datetime.now()
        },
        "typing_indicator": False,
        "last_calculation_parents": None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- تحميل الموارد المحسن ---
@st.cache_resource
def load_resources():
    """تحميل جميع الموارد اللازمة للوكيل الذكي."""
    resources = {
        "vector_db": None,
        "embedder": None,
        "model": None,
        "status": "initializing"
    }
    
    # تحميل قاعدة المتجهات
    if VECTOR_SEARCH_AVAILABLE:
        vector_db_path = "vector_db.pkl"
        if os.path.exists(vector_db_path):
            try:
                with open(vector_db_path, "rb") as f:
                    resources["vector_db"] = pickle.load(f)
                resources["embedder"] = SentenceTransformer(
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                st.success("✅ تم تحميل قاعدة المعرفة المتقدمة")
            except Exception as e:
                st.warning(f"⚠️ تعذر تحميل قاعدة المتجهات: {e}")
    
    # تهيئة نموذج الذكاء الاصطناعي
    if GEMINI_AVAILABLE and "GEMINI_API_KEY" in st.secrets:
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            resources["model"] = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 3000,
                    "top_p": 0.8,
                    "top_k": 40
                }
            )
            resources["status"] = "ready"
            st.success("🤖 تم تفعيل الوكيل الذكي بنجاح")
        except Exception as e:
            st.error(f"❌ فشل تفعيل الذكاء الاصطناعي: {e}")
            resources["status"] = "error"
    else:
        st.warning("⚠️ مفتاح API غير متوفر - الوضع التجريبي نشط")
        resources["status"] = "limited"
    
    return resources

# --- فئة الحاسبة الوراثية المحسنة ---
class AdvancedGeneticCalculator:
    """حاسبة وراثية متقدمة مع تفسير ذكي."""
    
    def __init__(self):
        self.calculation_history = []
    
    def describe_phenotype(self, genotype_dict: Dict[str, str]) -> Tuple[str, str]:
        """وصف النمط الظاهري من النمط الوراثي مع تفسير مفصل."""
        phenotypes = {gene: "" for gene in GENE_ORDER}
        
        # تحديد النمط الظاهري لكل جين
        for gene_name, gt_part in genotype_dict.items():
            alleles = gt_part.replace('•//', '').split('//')
            
            # البحث عن الأليل السائد
            for dominant_allele in GENE_DATA[gene_name]['dominance']:
                if dominant_allele in alleles:
                    phenotypes[gene_name] = GENE_DATA[gene_name]['alleles'][dominant_allele]['name']
                    break
        
        # تطبيق القواعد الخاصة
        if 'e//e' in genotype_dict.get('e', ''):
            phenotypes['B'] = 'أحمر متنحي'
            phenotypes['C'] = ''  # الأحمر المتنحي يخفي الأنماط
        
        if 'S' in genotype_dict.get('S', ''):
            if 'e//e' not in genotype_dict.get('e', ''):
                phenotypes['C'] = 'منتشر (سبريد)'
        
        # تحديد الجنس
        sex = "أنثى" if any('•' in genotype_dict.get(g, '') 
                           for g, d in GENE_DATA.items() 
                           if d['type_en'] == 'sex-linked') else "ذكر"
        
        # بناء الوصف النهائي
        desc_parts = [phenotypes.get('B')]
        if phenotypes.get('d') == 'مخفف':
            desc_parts.append('مخفف')
        if phenotypes.get('C'):
            desc_parts.append(phenotypes.get('C'))
        
        # بناء النمط الوراثي كنص
        gt_str_parts = [genotype_dict[gene].strip() for gene in GENE_ORDER]
        gt_str = " | ".join(gt_str_parts)
        
        final_phenotype = " ".join(filter(None, desc_parts))
        return f"{sex} {final_phenotype}", gt_str
    
    def calculate_advanced_genetics(self, parent_inputs: Dict) -> Dict:
        """حساب متقدم مع تحليل إحصائي ونصائح."""
        try:
            # بناء الأنماط الوراثية للوالدين
            parent_genotypes = {}
            
            for parent in ['male', 'female']:
                gt_parts = []
                for gene in GENE_ORDER:
                    gene_info = GENE_DATA[gene]
                    visible_name = parent_inputs[parent].get(f'{gene}_visible')
                    hidden_name = parent_inputs[parent].get(f'{gene}_hidden', visible_name)
                    
                    # تحويل الأسماء إلى رموز
                    wild_type_symbol = next((s for s, n in gene_info['alleles'].items() 
                                           if '+' in s or '⁺' in s), gene_info['dominance'][0])
                    
                    visible_symbol = NAME_TO_SYMBOL_MAP[gene].get(visible_name, wild_type_symbol)
                    hidden_symbol = NAME_TO_SYMBOL_MAP[gene].get(hidden_name, visible_symbol)
                    
                    # بناء النمط الوراثي
                    if gene_info['type_en'] == 'sex-linked' and parent == 'female':
                        gt_parts.append(f"•//{visible_symbol}")
                    else:
                        alleles = sorted([visible_symbol, hidden_symbol], 
                                       key=lambda x: gene_info['dominance'].index(x))
                        gt_parts.append(f"{alleles[0]}//{alleles[1]}")
                
                parent_genotypes[parent] = gt_parts
            
            # حساب الأمشاج
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
            
            # حساب النسل
            offspring_counts = collections.Counter()
            
            for m_gamete in male_gametes:
                for f_gamete in female_gametes:
                    son_dict, daughter_dict = {}, {}
                    
                    for i, gene in enumerate(GENE_ORDER):
                        alleles = sorted([m_gamete[i], f_gamete[i]], 
                                       key=lambda x: GENE_DATA[gene]['dominance'].index(x))
                        
                        if GENE_DATA[gene]['type_en'] == 'sex-linked':
                            son_dict[gene] = f"{alleles[0]}//{alleles[1]}"
                            daughter_dict[gene] = f"•//{m_gamete[i]}"
                        else:
                            gt_part = f"{alleles[0]}//{alleles[1]}"
                            son_dict[gene] = gt_part
                            daughter_dict[gene] = gt_part
                    
                    offspring_counts[self.describe_phenotype(son_dict)] += 1
                    offspring_counts[self.describe_phenotype(daughter_dict)] += 1
            
            # تحليل النتائج
            total = sum(offspring_counts.values())
            analysis = self._analyze_results(offspring_counts, total, parent_inputs)
            
            return {
                'results': offspring_counts,
                'total': total,
                'analysis': analysis,
                'parent_genotypes': parent_genotypes
            }
            
        except Exception as e:
            return {'error': f"خطأ في الحساب: {str(e)}"}
    
    def _analyze_results(self, results: Dict, total: int, parent_inputs: Dict) -> Dict:
        """تحليل متقدم للنتائج."""
        analysis = {
            'dominant_traits': [],
            'rare_combinations': [],
            'breeding_tips': [],
            'genetic_diversity': 0
        }
        
        # حساب التنوع الوراثي
        unique_combinations = len(results)
        analysis['genetic_diversity'] = (unique_combinations / total) * 100
        
        # تحديد الصفات السائدة
        for (phenotype, genotype), count in results.items():
            percentage = (count / total) * 100
            if percentage > 25:
                analysis['dominant_traits'].append({
                    'trait': phenotype,
                    'percentage': percentage
                })
            elif percentage < 10:
                analysis['rare_combinations'].append({
                    'trait': phenotype,
                    'percentage': percentage
                })
        
        # نصائح التربية
        if analysis['genetic_diversity'] > 80:
            analysis['breeding_tips'].append("تنوع وراثي ممتاز - مناسب لتحسين السلالة")
        elif analysis['genetic_diversity'] < 30:
            analysis['breeding_tips'].append("تنوع محدود - فكر في تنويع خطوط التربية")
        
        return analysis

# --- الوكيل الذكي المتقدم ---
class IntelligentGeneticAgent:
    """وكيل ذكي متقدم للمحادثة والحسابات الوراثية."""
    
    def __init__(self, resources: Dict):
        self.resources = resources
        self.calculator = AdvancedGeneticCalculator()
        self.conversation_memory = []
    
    def understand_query(self, query: str) -> Dict:
        """فهم متقدم لنية المستخدم."""
        query_lower = query.lower()
        
        intent = {
            'type': 'general',
            'confidence': 0.5,
            'entities': [],
            'calculation_needed': False,
            'genes_mentioned': []
        }
        
        # تحديد نوع الاستعلام
        if any(keyword in query_lower for keyword in ['احسب', 'حساب', 'نتائج', 'تزاوج', 'تربية']):
            intent['type'] = 'calculation'
            intent['calculation_needed'] = True
            intent['confidence'] = 0.9
            
        elif any(keyword in query_lower for keyword in ['شرح', 'وضح', 'كيف', 'ماذا', 'لماذا']):
            intent['type'] = 'explanation'
            intent['confidence'] = 0.8
            
        elif any(keyword in query_lower for keyword in ['مساعدة', 'help', 'مرحبا', 'السلام']):
            intent['type'] = 'greeting'
            intent['confidence'] = 0.9
        
        # استخراج الجينات المذكورة
        for gene, data in GENE_DATA.items():
            if any(allele['name'].lower() in query_lower for allele in data['alleles'].values()):
                intent['genes_mentioned'].append(gene)
        
        return intent
    
    def search_deep_memory(self, query: str, top_k: int = 5) -> List[Dict]:
        """البحث في الذاكرة العميقة المحسنة."""
        if not self.resources.get("vector_db") or not self.resources.get("embedder"):
            return []
        
        try:
            index = self.resources["vector_db"]["index"]
            chunks = self.resources["vector_db"]["chunks"]
            metadata = self.resources["vector_db"].get("metadata", [])
            
            query_embedding = self.resources["embedder"].encode([query])
            distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
            
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(chunks):
                    relevance_score = 1 / (1 + distance)
                    result = {
                        "type": "knowledge",
                        "content": chunks[idx],
                        "relevance": relevance_score,
                        "source": metadata[idx].get('source', 'قاعدة المعرفة') if idx < len(metadata) else 'قاعدة المعرفة',
                        "metadata": metadata[idx] if idx < len(metadata) else {}
                    }
                    results.append(result)
            
            return results
        except Exception as e:
            st.error(f"خطأ في البحث العميق: {e}")
            return []
    
    def generate_smart_response(self, query: str, intent: Dict) -> Dict:
        """توليد إجابة ذكية بناءً على السياق والنية."""
        
        if not self.resources.get("model"):
            return {
                "answer": "❌ عذراً، نظام الذكاء الاصطناعي غير متاح حالياً. يمكنك استخدام الحاسبة الوراثية مباشرة.",
                "sources": [],
                "calculation_widget": None
            }
        
        # البحث في المعرفة
        deep_results = self.search_deep_memory(query)
        context = "\n\n".join([f"معلومة: {r['content']}" for r in deep_results[:3]])
        
        # بناء المحفز الذكي
        system_prompt = """
        أنت 'العرّاب للجينات V6.0'، وكيل ذكاء اصطناعي متخصص في وراثة الحمام.
        
        شخصيتك:
        - خبير علمي ودود ومتحمس
        - تشرح المفاهيم بوضوح وبساطة
        - تستخدم الرموز التعبيرية بذكاء
        - تقدم أمثلة عملية
        - تربط المعلومات بالتطبيق العملي في التربية
        
        مهامك:
        1. الإجابة على الأسئلة العلمية بدقة
        2. شرح المفاهيم الوراثية المعقدة
        3. تقديم نصائح عملية للمربين
        4. اقتراح حسابات وراثية عند الحاجة
        """
        
        # تخصيص الاستجابة حسب النية
        if intent['type'] == 'calculation':
            user_prompt = f"""
            المستخدم يطلب حساباً وراثياً: "{query}"
            
            السياق المتاح: {context}
            
            قم بما يلي:
            1. اشرح ما سيتم حسابه
            2. اطلب المعلومات المطلوبة بوضوح
            3. وضح أهمية هذا الحساب للمربي
            4. اقترح استخدام الحاسبة المدمجة
            
            كن متحمساً ومشجعاً!
            """
            
        elif intent['type'] == 'explanation':
            user_prompt = f"""
            المستخدم يطلب شرحاً: "{query}"
            
            السياق المتاح: {context}
            الجينات المذكورة: {intent.get('genes_mentioned', [])}
            
            قم بشرح شامل ومبسط يتضمن:
            1. التعريف العلمي الدقيق
            2. أمثلة عملية من الحمام
            3. التأثير على التربية
            4. نصائح للمربين
            
            استخدم لغة علمية مبسطة ورموز تعبيرية مناسبة.
            """
            
        elif intent['type'] == 'greeting':
            user_prompt = f"""
            المستخدم يحييك: "{query}"
            
            رد بترحيب حار وودود، واعرض خدماتك:
            1. الإجابة على أسئلة الوراثة
            2. الحسابات الوراثية المتقدمة
            3. نصائح التربية
            4. شرح المفاهيم العلمية
            
            كن متحمساً ومرحباً!
            """
            
        else:
            user_prompt = f"""
            سؤال عام: "{query}"
            
            السياق المتاح: {context}
            
            أجب بطريقة علمية ودقيقة، مع التركيز على الجوانب العملية للتربية.
            """
        
        try:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.resources["model"].generate_content(full_prompt)
            
            # تحديد ما إذا كان يحتاج حاسبة
            needs_calculator = intent['calculation_needed'] or any(
                keyword in query.lower() for keyword in ['احسب', 'حساب', 'نتائج', 'تزاوج']
            )
            
            return {
                "answer": response.text,
                "sources": deep_results,
                "calculation_widget": needs_calculator,
                "intent": intent
            }
            
        except Exception as e:
            return {
                "answer": f"❌ عذراً، حدث خطأ في التوليد: {str(e)}",
                "sources": deep_results,
                "calculation_widget": intent['calculation_needed']
            }

# --- واجهة المحادثة العصرية ---
def render_chat_interface():
    """رسم واجهة المحادثة العصرية."""
    
    # حاوية المحادثة الرئيسية
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # شريط العنوان
    st.markdown('''
    <div class="header-bar">
        <div class="header-title">
            🧬 العرّاب للجينات V6.0
            <div class="status-indicator"></div>
        </div>
        <div style="font-size: 14px; opacity: 0.9;">
            وكيل ذكي متقدم • نشط الآن
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # منطقة المحادثة
    chat_container = st.container()
    
    with chat_container:
        st.markdown('<div class="chat-area">', unsafe_allow_html=True)
        
        # عرض الرسائل
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                st.markdown(f'''
                <div class="message user-message">
                    <div class="user-bubble">
                        {message["content"]}
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                
            else:  # assistant
                st.markdown(f'''
                <div class="message assistant-message">
                    <div class="avatar">🤖</div>
                    <div class="assistant-bubble">
                        {message["content"]}
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                
                # عرض الحاسبة المدمجة إذا كانت مطلوبة
                if message.get("show_calculator"):
                    render_embedded_calculator()
                
                # عرض المراجع
                if message.get("sources"):
                    with st.expander("📚 مصادر إضافية", expanded=False):
                        for source in message["sources"][:3]:
                            st.markdown(f"**{source.get('source', 'مصدر غير محدد')}** - درجة الصلة: {source.get('relevance', 0):.2f}")
                            st.markdown(f"_{source.get('content', '')[:200]}..._")
                            st.divider()
        
        # مؤشر الكتابة
        if st.session_state.get('typing_indicator', False):
            st.markdown('''
            <div class="typing-indicator">
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
                <span style="margin-left: 10px; color: #666;">العرّاب يفكر...</span>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # منطقة الإدخال
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    
    # أزرار التشغيل السريع
    st.markdown('<div class="quick-actions">', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("🧮 حساب وراثي", key="calc_btn"):
            handle_quick_action("أريد حساب نتائج التزاوج")
    
    with col2:
        if st.button("🎨 شرح الألوان", key="color_btn"):
            handle_quick_action("اشرح لي وراثة الألوان في الحمام")
    
    with col3:
        if st.button("📐 أنماط الريش", key="pattern_btn"):
            handle_quick_action("كيف تعمل وراثة أنماط الريش؟")
    
    with col4:
        if st.button("🔄 مثال عملي", key="example_btn"):
            handle_quick_action("أعطني مثال على تزاوج بين حمامتين")
    
    with col5:
        if st.button("💡 نصائح تربية", key="tips_btn"):
            handle_quick_action("ما هي أفضل نصائح لتربية الحمام؟")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # حقل الإدخال الرئيسي
    user_input = st.chat_input("اكتب سؤالك هنا... 💬", key="main_input")
    
    if user_input:
        handle_user_message(user_input)
    
    st.markdown('</div>', unsafe_allow_html=True)

def handle_quick_action(action_text: str):
    """معالجة الأزرار السريعة."""
    handle_user_message(action_text)

def handle_user_message(message: str):
    """معالجة رسالة المستخدم."""
    # إضافة رسالة المستخدم
    st.session_state.messages.append({"role": "user", "content": message})
    
    # تحديث الإحصائيات
    st.session_state.session_stats["queries"] += 1
    
    # تفعيل مؤشر الكتابة
    st.session_state.typing_indicator = True
    
    # إعادة تشغيل لإظهار التحديثات
    st.rerun()

def process_user_message(message: str, agent: IntelligentGeneticAgent):
    """معالجة رسالة المستخدم مع الوكيل الذكي."""
    
    # فهم النية
    intent = agent.understand_query(message)
    
    # توليد الاستجابة
    response_data = agent.generate_smart_response(message, intent)
    
    # إضافة رسالة المساعد
    assistant_message = {
        "role": "assistant",
        "content": response_data["answer"],
        "sources": response_data.get("sources", []),
        "show_calculator": response_data.get("calculation_widget", False),
        "intent": intent
    }
    
    st.session_state.messages.append(assistant_message)
    
    # إيقاف مؤشر الكتابة
    st.session_state.typing_indicator = False
    
    return response_data

# --- الحاسبة المدمجة العصرية ---
def render_embedded_calculator():
    """رسم الحاسبة الوراثية المدمجة في المحادثة."""
    
    st.markdown('''
    <div class="genetics-calculator">
        <div class="calc-header">
            🧮 الحاسبة الوراثية المدمجة
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns(2)
        
        parent_inputs = {'male': {}, 'female': {}}
        
        with col1:
            st.markdown("#### ♂️ **الذكر (الأب)**")
            
            for gene, data in GENE_DATA.items():
                with st.expander(f"{data['emoji']} {data['display_name_ar']}", expanded=False):
                    choices = ["(اختر الصفة)"] + [v['name'] for v in data['alleles'].values()]
                    
                    parent_inputs['male'][f'{gene}_visible'] = st.selectbox(
                        "الصفة الظاهرة:",
                        choices,
                        key=f"emb_male_{gene}_visible",
                        help=f"الصفة المرئية في {data['display_name_ar']}"
                    )
                    
                    parent_inputs['male'][f'{gene}_hidden'] = st.selectbox(
                        "الصفة المخفية:",
                        choices,
                        key=f"emb_male_{gene}_hidden",
                        help="الصفة غير المرئية (الأليل الثاني)"
                    )
        
        with col2:
            st.markdown("#### ♀️ **الأنثى (الأم)**")
            
            for gene, data in GENE_DATA.items():
                with st.expander(f"{data['emoji']} {data['display_name_ar']}", expanded=False):
                    choices = ["(اختر الصفة)"] + [v['name'] for v in data['alleles'].values()]
                    
                    parent_inputs['female'][f'{gene}_visible'] = st.selectbox(
                        "الصفة الظاهرة:",
                        choices,
                        key=f"emb_female_{gene}_visible",
                        help=f"الصفة المرئية في {data['display_name_ar']}"
                    )
                    
                    if data['type_en'] != 'sex-linked':
                        parent_inputs['female'][f'{gene}_hidden'] = st.selectbox(
                            "الصفة المخفية:",
                            choices,
                            key=f"emb_female_{gene}_hidden",
                            help="الصفة غير المرئية (الأليل الثاني)"
                        )
                    else:
                        st.info("الإناث لديها أليل واحد فقط للجينات المرتبطة بالجنس")
                        parent_inputs['female'][f'{gene}_hidden'] = parent_inputs['female'][f'{gene}_visible']
        
        st.markdown("---")
        
        # زر الحساب
        if st.button("🚀 احسب النتائج المتوقعة", use_container_width=True, type="primary"):
            if not all([
                parent_inputs['male'].get('B_visible') != "(اختر الصفة)",
                parent_inputs['female'].get('B_visible') != "(اختر الصفة)"
            ]):
                st.error("⚠️ يرجى اختيار اللون الأساسي لكلا الوالدين على الأقل")
            else:
                with st.spinner("🧬 جاري الحساب المتقدم..."):
                    calculator = AdvancedGeneticCalculator()
                    result_data = calculator.calculate_advanced_genetics(parent_inputs)
                    
                    if 'error' in result_data:
                        st.error(result_data['error'])
                    else:
                        display_advanced_results(result_data)
                        
                        # حفظ في الذاكرة للمحادثة
                        st.session_state.last_calculation_parents = parent_inputs
                        st.session_state.session_stats['calculations'] += 1

def display_advanced_results(result_data: Dict):
    """عرض النتائج المتقدمة بشكل تفاعلي."""
    
    results = result_data['results']
    total = result_data['total']
    analysis = result_data['analysis']
    
    st.markdown('''
    <div class="result-card">
        <h3>📊 النتائج المتوقعة</h3>
    </div>
    ''', unsafe_allow_html=True)
    
    # جدول النتائج
    df_results = pd.DataFrame([
        {
            'النمط الظاهري': phenotype,
            'النمط الوراثي': genotype,
            'العدد': count,
            'النسبة %': f"{(count/total)*100:.1f}%"
        }
        for (phenotype, genotype), count in results.items()
    ])
    
    st.dataframe(
        df_results,
        use_container_width=True,
        hide_index=True
    )
    
    # الرسم البياني
    col1, col2 = st.columns([2, 1])
    
    with col1:
        chart_data = df_results.set_index('النمط الظاهري')['النسبة %'].str.rstrip('%').astype('float')
        st.bar_chart(chart_data, height=300)
    
    with col2:
        st.metric("إجمالي التركيبات", total)
        st.metric("التنوع الوراثي", f"{analysis['genetic_diversity']:.1f}%")
        st.metric("الأنماط الفريدة", len(results))
    
    # التحليل المتقدم
    with st.expander("🔬 التحليل المتقدم", expanded=True):
        
        if analysis['dominant_traits']:
            st.markdown("**🔥 الصفات السائدة:**")
            for trait in analysis['dominant_traits']:
                st.markdown(f"- {trait['trait']}: {trait['percentage']:.1f}%")
        
        if analysis['rare_combinations']:
            st.markdown("**💎 التركيبات النادرة:**")
            for trait in analysis['rare_combinations']:
                st.markdown(f"- {trait['trait']}: {trait['percentage']:.1f}%")
        
        if analysis['breeding_tips']:
            st.markdown("**💡 نصائح التربية:**")
            for tip in analysis['breeding_tips']:
                st.info(tip)

# --- الواجهة الرئيسية المحسّنة ---
def main():
    """الواجهة الرئيسية للتطبيق المحسّن."""
    
    # تهيئة الجلسة
    initialize_session_state()
    
    # تحميل الموارد
    resources = load_resources()
    
    # إنشاء الوكيل الذكي
    agent = IntelligentGeneticAgent(resources)
    
    # إضافة رسالة ترحيب إذا لم تكن موجودة
    if not st.session_state.messages:
        welcome_message = """
        🧬 **مرحباً بك في العرّاب للجينات V6.0!**
        
        أنا وكيلك الذكي المتخصص في وراثة الحمام. يمكنني مساعدتك في:
        
        🔬 **الحسابات الوراثية المتقدمة** - احسب نتائج التزاوج بدقة علمية
        📚 **شرح المفاهيم الوراثية** - افهم كيف تعمل الوراثة بطريقة مبسطة  
        💡 **نصائح التربية العملية** - احصل على إرشادات من خبراء التربية
        🎨 **تحليل الألوان والأنماط** - اكتشف أسرار ألوان وأنماط الحمام
        
        **جرب الأزرار السريعة أدناه أو اكتب سؤالك مباشرة!** ✨
        """
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": welcome_message,
            "sources": [],
            "show_calculator": False
        })
    
    # رسم واجهة المحادثة
    render_chat_interface()
    
    # معالجة الرسائل الجديدة
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        last_message = st.session_state.messages[-1]["content"]
        
        # معالجة الرسالة مع الوكيل الذكي
        with st.spinner("🤖 العرّاب يحلل سؤالك..."):
            process_user_message(last_message, agent)
        
        # إعادة تشغيل لإظهار الاستجابة
        st.rerun()
    
    # شريط جانبي للإحصائيات والإعدادات
    with st.sidebar:
        st.markdown("### 📊 إحصائيات الجلسة")
        
        stats = st.session_state.session_stats
        st.metric("الاستعلامات", stats["queries"])
        st.metric("الحسابات", stats["calculations"])
        st.metric("البحث العميق", stats.get("deep_searches", 0))
        
        st.markdown("---")
        
        st.markdown("### ⚙️ الإعدادات")
        
        # إعدادات المستخدم
        detail_level = st.selectbox(
            "مستوى التفصيل:",
            ["بسيط", "متوسط", "متقدم"],
            index=1
        )
        
        show_formulas = st.checkbox("إظهار الصيغ الوراثية", value=True)
        
        # تحديث التفضيلات
        st.session_state.user_preferences.update({
            "detail_level": detail_level,
            "show_genetics_formulas": show_formulas
        })
        
        st.markdown("---")
        
        # معلومات النظام
        st.markdown("### ℹ️ معلومات النظام")
        st.markdown(f"**الإصدار:** V6.0")
        st.markdown(f"**الحالة:** {'🟢 نشط' if resources['status'] == 'ready' else '🟡 محدود'}")
        st.markdown(f"**الذكاء الاصطناعي:** {'✅ متاح' if GEMINI_AVAILABLE else '❌ غير متاح'}")
        
        if st.button("🔄 إعادة تشغيل الجلسة"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# --- تشغيل التطبيق ---
if __name__ == "__main__":
    main()
