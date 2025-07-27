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
        "conversation_context": [],  # جديد: لتتبع سياق المحادثة
        "agent_memory": {},  # جديد: ذاكرة الوكيل
        "user_preferences": {
            "max_results": 10,
            "analysis_depth": "متوسط",
            "language_style": "علمي",
            "include_charts": True,
            "show_trusted_sources": True,
            "conversation_mode": "تفاعلي",  # جديد
            "thinking_visibility": True,  # جديد
        },
        "session_stats": {
            "queries_count": 0,
            "successful_searches": 0,
            "charts_generated": 0,
            "calculations_performed": 0,
            "sources_referenced": 0,
            "deep_analyses": 0,  # جديد
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
            generation_config={"temperature": 0.2, "max_output_tokens": 6000})  # زيادة الحد الأقصى
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
        self.personality = {
            "name": "د. العرّاب الوراثي",
            "expertise": "علم الوراثة والألوان في الحمام",
            "approach": "علمي متفهم وتفاعلي",
            "knowledge_source": "مكتبة رقمية متخصصة"
        }
        
    def analyze_query_intent(self, query: str) -> Dict:
        """تحليل نية المستخدم من السؤال"""
        intent_analysis = {
            "type": "معلوماتي",  # معلوماتي، حسابي، استشاري، عام
            "complexity": "متوسط",  # بسيط، متوسط، معقد
            "requires_calculation": False,
            "requires_deep_analysis": False,
            "emotional_tone": "محايد",  # محايد، قلق، متحمس، محبط
        }
        
        # تحليل بسيط لنوع السؤال
        if any(word in query.lower() for word in ["احسب", "حساب", "نسبة", "احتمال"]):
            intent_analysis["type"] = "حسابي"
            intent_analysis["requires_calculation"] = True
            
        if any(word in query.lower() for word in ["لماذا", "كيف", "أسباب", "تفسير"]):
            intent_analysis["requires_deep_analysis"] = True
            intent_analysis["complexity"] = "معقد"
            
        if any(word in query.lower() for word in ["مصادر", "المراجع", "من أين"]):
            intent_analysis["type"] = "عام"
            
        return intent_analysis

    def generate_thinking_process(self, query: str, intent: Dict, search_results: List) -> str:
        """توليد عملية التفكير الظاهرة للمستخدم"""
        thinking_steps = []
        
        if intent["type"] == "عام":
            thinking_steps.append("🤔 أحلل طبيعة السؤال... يبدو أنك تسأل عن مصادري أو منهجيتي")
            thinking_steps.append("📚 سأوضح لك بدقة من أين أستمد معرفتي")
            
        elif intent["requires_deep_analysis"]:
            thinking_steps.append("🧠 أحلل السؤال بعمق... يتطلب تفسيراً علمياً شاملاً")
            thinking_steps.append("🔍 أبحث في مكتبتي الرقمية عن المعلومات ذات الصلة")
            if search_results:
                thinking_steps.append(f"📖 وجدت {len(search_results)} مرجع ذو صلة")
            thinking_steps.append("⚗️ سأربط النظريات العلمية بالتطبيق العملي")
            
        elif intent["type"] == "حسابي":
            thinking_steps.append("🧮 أتعرف على متطلبات الحساب الوراثي")
            thinking_steps.append("📊 سأطبق قوانين مندل وأسس الوراثة")
            
        else:
            thinking_steps.append("💭 أحلل استفسارك وأحدد أفضل طريقة للإجابة")
            if search_results:
                thinking_steps.append(f"📚 راجعت {len(search_results)} مرجع من مكتبتي")
        
        return " → ".join(thinking_steps)

    def create_advanced_prompt(self, query: str, context: str, intent: Dict, conversation_history: List) -> str:
        """إنشاء prompt متقدم ومتطور للوكيل"""
        
        # سياق المحادثة السابقة
        history_context = ""
        if conversation_history:
            recent_context = conversation_history[-3:]  # آخر 3 تبادلات
            history_context = "\n".join([f"سابق: {item['content'][:100]}..." for item in recent_context])
        
        # تخصيص الأسلوب حسب نوع السؤال
        if intent["type"] == "عام":
            style_instructions = """
أجب بوضوح تام عن هويتك ومصادرك. كن مباشراً وواثقاً.
استخدم عبارات مثل: "أنا د. العرّاب الوراثي"، "مكتبتي الرقمية تحتوي على"، "بناءً على المراجع المتخصصة لدي".
"""
        elif intent["requires_deep_analysis"]:
            style_instructions = """
قدم تحليلاً علمياً عميقاً ومفصلاً. استخدم الأمثلة والمقارنات.
اربط النظرية بالتطبيق العملي. كن تعليمياً وواضحاً.
استخدم التدرج المنطقي في الشرح من البسيط إلى المعقد.
"""
        else:
            style_instructions = """
أجب بأسلوب علمي مفهوم، مع الحفاظ على الدقة والوضوح.
استخدم أمثلة عملية من عالم تربية الحمام.
"""

        return f"""
أنت **د. العرّاب الوراثي** - خبير متخصص في علم الوراثة وألوان الحمام.

🧬 **هويتك العلمية:**
- اسمك: د. العرّاب الوراثي (نسخة 5.2)
- تخصصك: علم الوراثة التطبيقي في الحمام
- مصدر معرفتك: مكتبة رقمية متخصصة تحتوي على كتب ومراجع علمية موثقة
- منهجيتك: التحليل العلمي المبني على المصادر المعتمدة

📚 **مكتبتك الرقمية تشمل:**
- كتب الوراثة الكلاسيكية والحديثة
- أبحاث متخصصة في وراثة الطيور
- دراسات عن الألوان والأنماط في الحمام
- مراجع علمية محكمة في علم الوراثة التطبيقي

🎯 **السياق المرجعي من مكتبتك:**
```
{context}
```

💭 **سياق المحادثة السابقة:**
{history_context}

❓ **سؤال المستخدم الحالي:**
{query}

📋 **تعليمات الاستجابة:**
{style_instructions}

🔬 **منهجية الإجابة:**
1. **الثقة والوضوح:** أجب بثقة عالية مستمدة من خبرتك ومصادرك
2. **الأمانة العلمية:** التزم بما هو موجود في السياق المرجعي
3. **التحليل العميق:** فكر بعمق وحلل المعلومات قبل تقديم الإجابة
4. **التطبيق العملي:** اربط النظرية بالممارسة في تربية الحمام
5. **التفاعل الذكي:** تفاعل مع سياق المحادثة السابقة إذا كان ذا صلة

⚠️ **ضوابط مهمة:**
- إذا لم تجد المعلومة في السياق المرجعي، قل بوضوح: "هذه المعلومة غير متوفرة حالياً في مراجعي"
- لا تخترع معلومات غير موجودة في المصادر
- كن متفهماً ومساعداً في أسلوبك
- استخدم اللغة العربية الواضحة والدقيقة علمياً

**ابدأ إجابتك الآن:**
"""

    def process_query(self, query: str) -> Dict:
        """المعالجة الذكية الشاملة للاستفسار"""
        st.session_state.session_stats["queries_count"] += 1
        
        if not self.resources.get("model"):
            return {
                "answer": "❌ نظام الذكاء الاصطناعي غير متاح حالياً. يرجى التحقق من إعدادات API.",
                "confidence": 0.1,
                "thinking": "❌ لا يمكنني التفكير بدون نموذج الذكاء الاصطناعي"
            }

        # تحليل نية المستخدم
        intent = self.analyze_query_intent(query)
        
        # البحث في قاعدة المعرفة
        with st.spinner("🔍 أبحث في مكتبتي الرقمية..."):
            search_results = enhanced_search_knowledge(
                query, 
                self.resources, 
                top_k=self.preferences.get("max_results", 8)
            )
            
        # توليد عملية التفكير
        thinking_process = self.generate_thinking_process(query, intent, search_results)
        
        # إعداد السياق
        context_text = "\n\n---\n\n".join([r['content'] for r in search_results])
        conversation_history = st.session_state.get("conversation_context", [])
        
        # إنشاء الـ prompt المتقدم
        advanced_prompt = self.create_advanced_prompt(
            query, context_text, intent, conversation_history
        )
        
        # التحليل بالذكاء الاصطناعي
        with st.spinner("🧠 أحلل وأفكر بعمق..."):
            try:
                ai_response = self.resources["model"].generate_content(advanced_prompt)
                final_answer = ai_response.text
                
                # تحديث الذاكرة والسياق
                st.session_state.conversation_context.append({
                    "role": "user", "content": query
                })
                st.session_state.conversation_context.append({
                    "role": "assistant", "content": final_answer[:200] + "..."
                })
                
                # حساب الثقة
                confidence = self._calculate_confidence(search_results, intent)
                
                # تحديث الإحصائيات
                if intent["requires_deep_analysis"]:
                    st.session_state.session_stats["deep_analyses"] += 1
                if search_results:
                    st.session_state.session_stats["successful_searches"] += 1
                    st.session_state.session_stats["sources_referenced"] += len(search_results)
                
                return {
                    "answer": final_answer,
                    "confidence": confidence,
                    "thinking": thinking_process,
                    "sources": search_results,
                    "intent": intent
                }
                
            except Exception as e:
                return {
                    "answer": f"❌ عذراً، واجهت صعوبة في معالجة استفسارك: {str(e)}",
                    "confidence": 0.2,
                    "thinking": "❌ حدث خطأ أثناء التحليل",
                    "sources": search_results,
                    "intent": intent
                }
    
    def _calculate_confidence(self, search_results: List, intent: Dict) -> float:
        """حساب درجة الثقة في الإجابة"""
        base_confidence = 0.7
        
        if search_results:
            # متوسط نقاط البحث
            avg_score = np.mean([r['score'] for r in search_results])
            base_confidence += avg_score * 0.2
            
        # تعديل حسب نوع السؤال
        if intent["type"] == "عام":
            base_confidence += 0.1  # أكثر ثقة في الأسئلة العامة
        elif intent["requires_deep_analysis"]:
            base_confidence -= 0.1  # أقل ثقة في التحليلات المعقدة
            
        return min(0.95, max(0.3, base_confidence))

# --- 9. واجهة المستخدم الشاملة ---
def main():
    initialize_session_state()
    resources = load_enhanced_resources()
    model = initialize_enhanced_gemini()
    resources["model"] = model
    
    # إنشاء الوكيل الذكي الجديد
    agent = IntelligentGeneticsAgent(resources, st.session_state.user_preferences)

    st.markdown('<div class="main-header"><h1>🚀 العرّاب للجينات V5.2</h1><p><strong>النسخة الذكية المتطورة - وكيل بشخصية علمية محسنة</strong></p></div>', unsafe_allow_html=True)
    
    # عرض إحصائيات الجلسة في الشريط الجانبي
    with st.sidebar:
        st.markdown("### 📊 إحصائيات الجلسة")
        stats = st.session_state.session_stats
        st.metric("🔍 الاستفسارات", stats["queries_count"])
        st.metric("📚 البحوث الناجحة", stats["successful_searches"])
        st.metric("🧠 التحليلات العميقة", stats["deep_analyses"])
        st.metric("📖 المصادر المرجعية", stats["sources_referenced"])
        
        st.markdown("---")
        st.markdown("### 🤖 حالة الوكيل")
        if resources.get("model"):
            st.success("✅ الوكيل جاهز ونشط")
        else:
            st.error("❌ الوكيل غير متاح")
            
        if resources.get("vector_db"):
            st.success("✅ قاعدة المعرفة محملة")
        else:
            st.warning("⚠️ قاعدة المعرفة غير متاحة")
    
    tab1, tab2, tab3, tab4 = st.tabs(["💬 المحادثة الذكية", "🧬 الحاسبة الوراثية", "🧠 ذاكرة الوكيل", "⚙️ الإعدادات"])

    with tab1:
        st.subheader("🤖 تحدث مع د. العرّاب الوراثي")
        
        # خيارات المحادثة
        col1, col2 = st.columns([3, 1])
        with col2:
            show_thinking = st.toggle("🧠 إظهار التفكير", value=st.session_state.user_preferences["thinking_visibility"])
            st.session_state.user_preferences["thinking_visibility"] = show_thinking
        
        # منطقة المحادثة
        chat_container = st.container(height=500)
        
        # عرض المحادثات السابقة
        for message in st.session_state.messages:
            with chat_container.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # عرض عملية التفكير إذا كانت متاحة
                if message["role"] == "assistant" and "thinking" in message and show_thinking:
                    with st.expander("🧠 كيف فكرت في هذا؟", expanded=False):
                        st.markdown(f'<div class="agent-thinking">💭 {message["thinking"]}</div>', unsafe_allow_html=True)
                
                # عرض المصادر إذا كانت متاحة
                if message["role"] == "assistant" and "sources" in message and message["sources"]:
                    with st.expander(f"📚 المصادر المرجعية ({len(message['sources'])})", expanded=False):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f'<div class="trusted-source"><strong>مرجع {i}:</strong><br>{source["content"][:300]}...</div>', unsafe_allow_html=True)
        
        # مربع الإدخال
        if prompt := st.chat_input("مثال: ما هي مصادرك؟ أو: اشرح لي وراثة اللون الأزرق"):
            # إضافة رسالة المستخدم
            st.session_state.messages.append({"role": "user", "content": prompt})
            chat_container.chat_message("user").markdown(prompt)
            
            # معالجة الاستفسار بالوكيل الجديد
            with chat_container.chat_message("assistant"):
                with st.spinner("🤔 د. العرّاب يفكر..."):
                    response_data = agent.process_query(prompt)
                
                # عرض عملية التفكير
                if show_thinking and response_data.get("thinking"):
                    st.markdown(f'<div class="agent-thinking">💭 {response_data["thinking"]}</div>', unsafe_allow_html=True)
                
                # عرض الإجابة
                st.markdown(response_data["answer"])
                
                # عرض درجة الثقة
                confidence = response_data.get("confidence", 0.5)
                confidence_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
                st.caption(f"{confidence_color} درجة الثقة: {confidence:.1%}")
                
                # حفظ الرسالة مع البيانات الإضافية
                assistant_message = {
                    "role": "assistant", 
                    "content": response_data["answer"],
                    "thinking": response_data.get("thinking", ""),
                    "sources": response_data.get("sources", []),
                    "confidence": confidence,
                    "intent": response_data.get("intent", {})
                }
                st.session_state.messages.append(assistant_message)

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

            col_calc1, col_calc2 = st.columns([2, 1])
            with col_calc1:
                calculate_btn = st.button("🚀 احسب النتائج", use_container_width=True, type="primary")
            with col_calc2:
                ask_agent_calc = st.button("🤖 اسأل الوكيل", use_container_width=True)

            if calculate_btn:
                if not all(val != "(اختر)" for val in [parent_inputs['male']['B_visible'], parent_inputs['female']['B_visible']]):
                    st.error("⚠️ يرجى اختيار اللون الأساسي لكلا الوالدين.")
                else:
                    calculator = EnhancedGeneticCalculator()
                    result_data = calculator.calculate_detailed_genetics(parent_inputs)
                    st.session_state.calculation_history.append(result_data)
                    st.session_state.session_stats["calculations_performed"] += 1

            if ask_agent_calc:
                if not all(val != "(اختر)" for val in [parent_inputs['male']['B_visible'], parent_inputs['female']['B_visible']]):
                    st.error("⚠️ يرجى اختيار اللون الأساسي لكلا الوالدين أولاً.")
                else:
                    # تكوين سؤال للوكيل حول الحساب
                    calc_query = f"اشرح لي نتائج تزاوج ذكر {parent_inputs['male']['B_visible']} مع أنثى {parent_inputs['female']['B_visible']}"
                    
                    # إضافة السؤال إلى المحادثة
                    st.session_state.messages.append({"role": "user", "content": calc_query})
                    
                    # معالجة بالوكيل
                    response_data = agent.process_query(calc_query)
                    assistant_message = {
                        "role": "assistant", 
                        "content": response_data["answer"],
                        "thinking": response_data.get("thinking", ""),
                        "sources": response_data.get("sources", []),
                        "confidence": response_data.get("confidence", 0.5)
                    }
                    st.session_state.messages.append(assistant_message)
                    
                    st.success("✅ تم إضافة السؤال إلى المحادثة. انتقل إلى تبويب المحادثة لرؤية الإجابة.")

        # عرض آخر نتيجة حساب
        if st.session_state.calculation_history:
            last_calc = st.session_state.calculation_history[-1]
            st.subheader("📊 أحدث النتائج")
            if 'error' in last_calc:
                st.error(last_calc['error'])
            else:
                df_results = pd.DataFrame([{
                    'النمط الظاهري': p, 
                    'النمط الوراثي': g, 
                    'العدد': c,
                    'النسبة %': f"{(c/last_calc['total_offspring'])*100:.1f}%"
                } for (p, g), c in last_calc['results'].items()])
                
                st.dataframe(df_results, use_container_width=True)
                
                # رسم بياني للتوزيع
                if st.session_state.user_preferences["include_charts"]:
                    col_chart1, col_chart2 = st.columns(2)
                    
                    with col_chart1:
                        fig_sex = px.pie(
                            values=list(last_calc['sex_distribution'].values()), 
                            names=list(last_calc['sex_distribution'].keys()), 
                            title="توزيع الجنس"
                        )
                        st.plotly_chart(fig_sex, use_container_width=True)
                    
                    with col_chart2:
                        phenotype_data = {}
                        for (phenotype, genotype), count in last_calc['results'].items():
                            if phenotype in phenotype_data:
                                phenotype_data[phenotype] += count
                            else:
                                phenotype_data[phenotype] = count
                        
                        fig_pheno = px.bar(
                            x=list(phenotype_data.keys()),
                            y=list(phenotype_data.values()),
                            title="توزيع الأنماط الظاهرية"
                        )
                        st.plotly_chart(fig_pheno, use_container_width=True)
                    
                    st.session_state.session_stats["charts_generated"] += 2

    with tab3:
        st.subheader("🧠 ذاكرة الوكيل وسياق المحادثة")
        
        col_mem1, col_mem2 = st.columns(2)
        
        with col_mem1:
            st.markdown("#### 💭 سياق المحادثة الأخيرة")
            if st.session_state.conversation_context:
                for i, context_item in enumerate(st.session_state.conversation_context[-6:]):  # آخر 6 عناصر
                    role_icon = "🧑‍💼" if context_item["role"] == "user" else "🤖"
                    st.markdown(f"{role_icon} **{context_item['role']}:** {context_item['content']}")
            else:
                st.info("لا يوجد سياق محادثة بعد.")
        
        with col_mem2:
            st.markdown("#### 📈 تحليل أنماط الاستفسارات")
            if st.session_state.messages:
                user_messages = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]
                
                # تحليل بسيط لأنواع الأسئلة
                question_types = {
                    "أسئلة عامة": sum(1 for msg in user_messages if any(word in msg.lower() for word in ["ما", "من", "أين", "مصادر"])),
                    "أسئلة تحليلية": sum(1 for msg in user_messages if any(word in msg.lower() for word in ["كيف", "لماذا", "اشرح"])),
                    "أسئلة حسابية": sum(1 for msg in user_messages if any(word in msg.lower() for word in ["احسب", "نسبة", "احتمال"])),
                }
                
                for q_type, count in question_types.items():
                    st.metric(q_type, count)
            else:
                st.info("لا توجد بيانات للتحليل بعد.")
        
        # زر مسح الذاكرة
        if st.button("🗑️ مسح ذاكرة الوكيل", type="secondary"):
            st.session_state.conversation_context = []
            st.session_state.agent_memory = {}
            st.success("✅ تم مسح ذاكرة الوكيل.")

    with tab4:
        st.subheader("⚙️ الإعدادات المتقدمة")
        
        col_settings1, col_settings2 = st.columns(2)
        
        with col_settings1:
            st.markdown("#### 🔧 إعدادات البحث والتحليل")
            prefs = st.session_state.user_preferences
            
            prefs['max_results'] = st.slider("عدد نتائج البحث", 5, 15, prefs['max_results'])
            prefs['analysis_depth'] = st.select_slider("عمق التحليل", ["بسيط", "متوسط", "عميق"], value=prefs['analysis_depth'])
            prefs['language_style'] = st.radio("أسلوب اللغة", ["علمي", "مبسط", "تقني"], index=["علمي", "مبسط", "تقني"].index(prefs['language_style']))
            prefs['conversation_mode'] = st.selectbox("نمط المحادثة", ["تفاعلي", "رسمي", "ودود"], index=["تفاعلي", "رسمي", "ودود"].index(prefs['conversation_mode']))
            
        with col_settings2:
            st.markdown("#### 🎨 إعدادات العرض")
            prefs['include_charts'] = st.toggle("تضمين الرسوم البيانية", value=prefs['include_charts'])
            prefs['show_trusted_sources'] = st.toggle("إظهار المصادر الموثقة", value=prefs['show_trusted_sources'])
            prefs['thinking_visibility'] = st.toggle("إظهار عملية التفكير", value=prefs['thinking_visibility'])
            
            st.markdown("#### 📊 إحصائيات الجلسة الكاملة")
            stats_df = pd.DataFrame([
                {"المقياس": "الاستفسارات الكلية", "القيمة": stats["queries_count"]},
                {"المقياس": "البحوث الناجحة", "القيمة": stats["successful_searches"]},
                {"المقياس": "التحليلات العميقة", "القيمة": stats["deep_analyses"]},
                {"المقياس": "الحسابات المنجزة", "القيمة": stats["calculations_performed"]},
                {"المقياس": "المصادر المراجعة", "القيمة": stats["sources_referenced"]},
                {"المقياس": "الرسوم المولدة", "القيمة": stats["charts_generated"]},
            ])
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # أزرار الإدارة
        col_admin1, col_admin2, col_admin3 = st.columns(3)
        
        with col_admin1:
            if st.button("🔄 إعادة تعيين الإعدادات", use_container_width=True):
                st.session_state.user_preferences = {
                    "max_results": 10,
                    "analysis_depth": "متوسط",
                    "language_style": "علمي",
                    "include_charts": True,
                    "show_trusted_sources": True,
                    "conversation_mode": "تفاعلي",
                    "thinking_visibility": True,
                }
                st.success("✅ تم إعادة تعيين الإعدادات.")
        
        with col_admin2:
            if st.button("🗑️ مسح تاريخ المحادثة", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_context = []
                st.success("✅ تم مسح تاريخ المحادثة.")
        
        with col_admin3:
            if st.button("📊 إعادة تعيين الإحصائيات", use_container_width=True):
                st.session_state.session_stats = {
                    "queries_count": 0,
                    "successful_searches": 0,
                    "charts_generated": 0,
                    "calculations_performed": 0,
                    "sources_referenced": 0,
                    "deep_analyses": 0,
                }
                st.success("✅ تم إعادة تعيين الإحصائيات.")

        # معلومات النظام
        with st.expander("ℹ️ معلومات النظام والمصادر"):
            st.markdown("""
            ### 🔬 نبذة عن د. العرّاب الوراثي V5.2
            
            **الهوية العلمية:**
            - خبير متخصص في علم الوراثة وألوان الحمام
            - يعتمد على مكتبة رقمية متخصصة ومتقدمة
            - يستخدم منهجية علمية في التحليل والاستنتاج
            
            **القدرات المتطورة:**
            - تحليل ذكي لنوايا المستخدم
            - عملية تفكير شفافة وقابلة للعرض
            - ذاكرة محادثة متقدمة
            - تكامل بين النظرية والتطبيق العملي
            
            **المصادر والمراجع:**
            - كتب الوراثة الكلاسيكية والحديثة
            - أبحاث متخصصة في وراثة الطيور
            - دراسات علمية محكمة
            - مراجع تطبيقية في تربية الحمام
            """)

if __name__ == "__main__":
    main()
