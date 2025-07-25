# ===================================================================
# 🚀 العرّاب للجينات - Ultra Fast Version (سرعة ChatGPT)
# تحسينات جذرية: إزالة Langchain، استخدام Gemini مباشرة، ذاكرة محسنة
# ===================================================================

import streamlit as st
from itertools import product
import collections
import pandas as pd
import google.generativeai as genai
import json
import os
import time
from datetime import datetime
import hashlib

# --- 1. إعدادات الصفحة ---
st.set_page_config(layout="wide", page_title="العرّاب للجينات - السرعة الفائقة")

# --- 2. قاعدة البيانات الوراثية ---
GENE_DATA = {
    'B': {
        'display_name_ar': "اللون الأساسي", 'type_en': 'sex-linked',
        'alleles': {
            'BA': {'name': 'آش ريد', 'is_recessive': False},
            '+': {'name': 'أزرق/أسود', 'is_recessive': False},
            'b': {'name': 'بني', 'is_recessive': True}
        },
        'dominance': ['BA', '+', 'b']
    },
    'd': {
        'display_name_ar': "التخفيف", 'type_en': 'sex-linked',
        'alleles': {
            '+': {'name': 'عادي (غير مخفف)', 'is_recessive': False},
            'd': {'name': 'مخفف', 'is_recessive': True}
        },
        'dominance': ['+', 'd']
    },
    'e': {
        'display_name_ar': "أحمر متنحي", 'type_en': 'autosomal',
        'alleles': {
            '+': {'name': 'عادي (غير أحمر متنحي)', 'is_recessive': False},
            'e': {'name': 'أحمر متنحي', 'is_recessive': True}
        },
        'dominance': ['+', 'e']
    },
    'C': {
        'display_name_ar': "النمط", 'type_en': 'autosomal',
        'alleles': {
            'CT': {'name': 'نمط تي (مخملي)', 'is_recessive': False},
            'C': {'name': 'تشيكر', 'is_recessive': False},
            '+': {'name': 'بار (شريط)', 'is_recessive': False},
            'c': {'name': 'بدون شريط', 'is_recessive': True}
        },
        'dominance': ['CT', 'C', '+', 'c']
    },
    'S': {
        'display_name_ar': "الانتشار (سبريد)", 'type_en': 'autosomal',
        'alleles': {
            'S': {'name': 'منتشر (سبريد)', 'is_recessive': False},
            '+': {'name': 'عادي (غير منتشر)', 'is_recessive': False}
        },
        'dominance': ['S', '+']
    }
}
GENE_ORDER = list(GENE_DATA.keys())
NAME_TO_SYMBOL_MAP = {
    gene: {info['name']: symbol for symbol, info in data['alleles'].items()}
    for gene, data in GENE_DATA.items()
}

# --- 3. المحرك الوراثي ---
class GeneticCalculator:
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
        if phenotypes.get('C'): desc_parts.append(phenotypes.get('C'))
        gt_str_parts = [genotype_dict[gene].strip() for gene in GENE_ORDER]
        gt_str = " | ".join(gt_str_parts)
        final_phenotype = " ".join(filter(None, desc_parts))
        return f"{sex} {final_phenotype}", gt_str

def predict_genetics_final(parent_inputs):
    calculator = GeneticCalculator()
    parent_genotypes = {}
    for parent in ['male', 'female']:
        gt_parts = []
        for gene in GENE_ORDER:
            gene_info = GENE_DATA[gene]
            visible_name = parent_inputs[parent].get(f'{gene}_visible')
            hidden_name = parent_inputs[parent].get(f'{gene}_hidden', visible_name)
            wild_type_symbol = next((s for s, n in gene_info['alleles'].items() if '+' in s or '⁺' in s), gene_info['dominance'][0])
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
                    son_dict[gene], daughter_dict[gene] = f"{alleles[0]}//{alleles[1]}", f"•//{m_gamete[i]}"
                else:
                    gt_part = f"{alleles[0]}//{alleles[1]}"
                    son_dict[gene], daughter_dict[gene] = gt_part, gt_part
            offspring_counts[calculator.describe_phenotype(son_dict)] += 1
            offspring_counts[calculator.describe_phenotype(daughter_dict)] += 1
    return offspring_counts

# --- 4. قاعدة المعرفة الفائقة السرعة ---
ULTRA_FAST_KNOWLEDGE = {
    'الألوان الأساسية': """
🎨 **الألوان الأساسية في الحمام الزاجل:**

**1. الآش ريد (Ash Red) - BA:**
- لون أحمر مائل للرمادي مع بريق معدني
- سائد على جميع الألوان الأخرى
- يظهر بوضوح في الذكور والإناث
- رمز الجين: BA

**2. الأزرق/الأسود (Blue/Black) - +:**
- اللون الطبيعي الأساسي للحمام البري
- أزرق رمادي مع شريطين أسودين على الأجنحة
- متوسط السيادة
- رمز الجين: + (الطبيعي)

**3. البني (Brown/Red) - b:**
- لون بني محمر أو شوكولاتي
- أكثر الألوان تنحياً
- يحتاج جينين متنحيين (bb) ليظهر
- رمز الجين: b

**الوراثة:** هذه الألوان تتحكم فيها الجينات المرتبطة بالجنس على الكروموسوم Z.

**ترتيب السيادة:** BA > + > b
    """,
    'جين الانتشار': """
🔸 **جين الانتشار (Spread Gene - S):**

**الوظيفة:**
- ينشر اللون الأساسي على كامل الريشة
- يخفي جميع الأنماط (البار، التشيكر، المخملي)
- يعطي لوناً موحداً بدون خطوط أو نقاط

**خصائص الوراثة:**
- جين جسمي (autosomal) سائد
- يحتاج جين واحد فقط ليظهر (S+)
- موقعه على كروموسوم غير جنسي

**التأثير العملي:**
- حمامة زرقاء عادية + جين الانتشار = زرقاء موحدة
- حمامة آش ريد + جين الانتشار = حمراء موحدة
- حمامة بنية + جين الانتشار = بنية موحدة

**أمثلة:**
- SS أو S+ = انتشار كامل
- ++ = بدون انتشار (الأنماط تظهر)
    """,
    'الوراثة المرتبطة بالجنس': """
♂️♀️ **الوراثة المرتبطة بالجنس في الحمام:**

**نظام الكروموسومات:**
- الذكور: ZZ (متماثلان)
- الإناث: ZW (مختلفان)

**في الذكور (ZZ):**
- لديهم نسختان من كل جين مرتبط بالجنس
- يمكن أن يكونوا حاملين للصفات المتنحية
- النمط الوراثي: gene1//gene2
- مثال: BA//b (آش ريد يحمل بني)

**في الإناث (ZW):**
- لديهن نسخة واحدة فقط من الجينات المرتبطة بالجنس
- لا يمكن أن يكن حاملات
- ما يحملنه يظهر مباشرة
- النمط الوراثي: •//gene
- مثال: •//BA (أنثى آش ريد)

**الجينات المرتبطة بالجنس:**
1. **اللون الأساسي (B):** آش ريد، أزرق، بني
2. **التخفيف (d):** عادي أو مخفف

**قانون الوراثة:**
- الأب يعطي الإناث كروموسوم Z واحد
- الأم تعطي الذكور كروموسوم Z والإناث كروموسوم W
    """,
    'أنماط الريش': """
🪶 **أنماط الريش (Pattern Gene - C):**

**1. نمط T المخملي (Velvet) - CT:**
- الأقوى سيادة بين جميع الأنماط
- لون موحد مخملي بدون أي خطوط
- يشبه تأثير جين الانتشار لكنه مختلف
- رمز: CT

**2. التشيكر (Checker) - C:**
- نقاط أو مربعات صغيرة منتظمة
- سائد على البار والعادي
- يظهر بوضوح على الأجنحة
- رمز: C

**3. البار (Bar) - +:**
- خطوط عرضية داكنة على الأجنحة
- النمط الطبيعي في الحمام البري
- عادة خطان واضحان
- رمز: + (الطبيعي)

**4. بدون نمط (No Pattern) - c:**
- الأكثر تنحياً
- لون موحد بدون أي خطوط أو نقاط
- يحتاج جينين متنحيين (cc)
- رمز: c

**ترتيب السيادة:** CT > C > + > c

**الوراثة:** جين جسمي (autosomal) - غير مرتبط بالجنس

**أمثلة عملية:**
- CT/+ = مخملي
- C/+ = تشيكر
- +/+ = بار
- c/c = بدون نمط
    """,
    'التخفيف': """
💧 **جين التخفيف (Dilution Gene - d):**

**الوظيفة:**
- يخفف كثافة اللون الأساسي
- يجعل الألوان أفتح وأقل كثافة
- مرتبط بالجنس (على كروموسوم Z)

**التأثير على الألوان:**
- **آش ريد مخفف:** أصفر ذهبي أو كريمي
- **أزرق مخفف:** فضي أو رمادي فاتح
- **بني مخفف:** أصفر باهت أو كريمي

**الوراثة:**
- جين متنحي مرتبط بالجنس
- رمز الجين العادي: +
- رمز الجين المخفف: d

**في الذكور:**
- +/+ = لون عادي
- +/d = لون عادي (حامل)
- d/d = لون مخفف

**في الإناث:**
- •/+ = لون عادي
- •/d = لون مخفف

**ملاحظة مهمة:**
- الذكور يحتاجون جينين مخففين ليظهر التخفيف
- الإناث تحتاج جين واحد فقط
    """,
    'أحمر متنحي': """
🔴 **الأحمر المتنحي (Recessive Red - e):**

**الطبيعة:**
- جين جسمي متنحي قوي جداً
- يخفي جميع الألوان الأساسية الأخرى
- يعطي لوناً أحمر موحداً

**الشروط للظهور:**
- يحتاج جينين متنحيين (e/e)
- يظهر في الذكور والإناث بنفس الطريقة
- رمز الجين العادي: +
- رمز الأحمر المتنحي: e

**التأثير:**
- عندما يكون e/e:
  - يخفي لون الآش ريد
  - يخفي اللون الأزرق
  - يخفي اللون البني
  - يعطي أحمر موحد

**التفاعل مع الجينات الأخرى:**
- يلغي تأثير أنماط الريش
- لا يتأثر بجين الانتشار
- يمكن أن يتأثر بالتخفيف (يصبح أصفر)

**أمثلة:**
- +/+ = عادي (الألوان الأساسية تظهر)
- +/e = عادي (حامل)
- e/e = أحمر متنحي

**نصيحة للمربين:**
- صعب التنبؤ به لأنه متنحي
- يمكن أن يظهر فجأة من والدين عاديين
- مفيد لإنتاج ألوان موحدة
    """
}

# --- 5. المحرك الذكي الفائق السرعة ---
class UltraFastAI:
    def __init__(self):
        if "GEMINI_API_KEY" not in st.secrets:
            self.model = None
            self.error = "مفتاح API غير موجود"
        else:
            try:
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                self.model = genai.GenerativeModel(
                    'gemini-1.5-flash',
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=1000,
                        top_p=0.8,
                        top_k=10
                    )
                )
                self.error = None
            except Exception as e:
                self.model = None
                self.error = str(e)
    
    def get_instant_answer(self, query):
        query_lower = query.lower().strip()
        
        keywords_map = {
            'الألوان الأساسية': ['ألوان أساسية', 'الوان اساسية', 'آش ريد', 'أزرق', 'بني', 'ash red', 'blue', 'brown', 'basic colors'],
            'جين الانتشار': ['انتشار', 'سبريد', 'spread', 'منتشر'],
            'الوراثة المرتبطة بالجنس': ['مرتبط بالجنس', 'مرتبطة بالجنس', 'sex-linked', 'sex linked', 'جنسية', 'ذكر', 'أنثى'],
            'أنماط الريش': ['أنماط', 'انماط', 'نمط', 'pattern', 'بار', 'تشيكر', 'checker', 'bar', 'مخملي', 'velvet'],
            'التخفيف': ['تخفيف', 'مخفف', 'dilution', 'dilute', 'فاتح', 'باهت'],
            'أحمر متنحي': ['أحمر متنحي', 'احمر متنحي', 'recessive red', 'أحمر', 'احمر']
        }
        
        for topic, keywords in keywords_map.items():
            if any(keyword in query_lower for keyword in keywords):
                return ULTRA_FAST_KNOWLEDGE[topic]
        
        return None
    
    def get_smart_answer(self, query, context_history=""):
        if not self.model:
            return f"❌ خطأ في النظام: {self.error}"
        
        try:
            prompt = f"""
أنت خبير في علم وراثة الحمام الزاجل. أجب بإيجاز ووضوح.

{context_history}

السؤال: {query}

قواعد الإجابة:
1. إجابة مختصرة ومفيدة (أقل من 300 كلمة)
2. باللغة العربية
3. معلومات دقيقة فقط
4. استخدم الرموز والأمثلة العملية
5. إذا لم تعرف الإجابة، قل ذلك بوضوح

الإجابة:
            """
            
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                return f"🧠 **إجابة ذكية:**\n\n{response.text}"
            else:
                return "❌ لم أتمكن من إنتاج إجابة. جرب إعادة صياغة السؤال."
                
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower():
                return "📊 تم تجاوز حد الاستخدام اليومي. جرب غداً."
            elif "blocked" in error_msg.lower():
                return "🚫 تم حجب الاستعلام. جرب صياغة مختلفة."
            else:
                return f"⚠️ خطأ مؤقت: {error_msg[:100]}... جرب مرة أخرى."

# --- 6. نظام الذاكرة المحسن ---
class FastMemory:
    def __init__(self):
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'session_stats' not in st.session_state:
            st.session_state.session_stats = {
                'questions': 0,
                'calculations': 0,
                'start_time': datetime.now()
            }
    
    def add_exchange(self, question, answer, response_time):
        exchange = {
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'question': question,
            'answer': answer,
            'response_time': response_time,
            'id': hashlib.md5(f"{question}{datetime.now()}".encode()).hexdigest()[:8]
        }
        
        st.session_state.chat_history.append(exchange)
        st.session_state.session_stats['questions'] += 1
        
        if len(st.session_state.chat_history) > 15:
            st.session_state.chat_history = st.session_state.chat_history[-15:]
    
    def get_context(self, last_n=3):
        if not st.session_state.chat_history:
            return ""
        
        recent = st.session_state.chat_history[-last_n:]
        context = "السياق من المحادثة الأخيرة:\n"
        
        for exchange in recent:
            context += f"س: {exchange['question'][:80]}...\n"
            context += f"ج: {exchange['answer'][:120]}...\n---\n"
        
        return context

# --- 7. إنشاء كائنات النظام ---
fast_ai = UltraFastAI()
memory = FastMemory()

# --- 8. واجهة التطبيق الفائقة ---
st.title("🚀 العرّاب للجينات - السرعة الفائقة")

status_col1, status_col2, status_col3 = st.columns(3)
with status_col1:
    if fast_ai.model:
        st.success("🟢 الذكاء الاصطناعي جاهز")
    else:
        st.error("🔴 الذكاء الاصطناعي غير متاح")

with status_col2:
    avg_response = sum([ex.get('response_time', 0) for ex in st.session_state.chat_history[-5:]]) / max(len(st.session_state.chat_history[-5:]), 1)
    st.metric("⚡ متوسط الاستجابة", f"{avg_response:.1f}s")

with status_col3:
    st.metric("💬 أسئلة الجلسة", st.session_state.session_stats['questions'])

tab1, tab2, tab3 = st.tabs(["🤖 المحادثة السريعة", "🧬 الحاسبة الوراثية", "📊 إحصائيات الأداء"])

with tab1:
    st.header("💬 تحدث مع الخبير - سرعة ChatGPT")
    
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.chat_history:
            st.subheader("📜 المحادثة الحالية")
            
            for exchange in st.session_state.chat_history[-5:]:
                st.markdown(f"""
                <div style="background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px 0;">
                    <strong>👤 أنت [{exchange['timestamp']}]:</strong><br>
                    {exchange['question']}
                </div>
                """, unsafe_allow_html=True)
                
                response_color = "#e8f5e8" if exchange['response_time'] < 3 else "#fff3e0"
                st.markdown(f"""
                <div style="background-color: {response_color}; padding: 10px; border-radius: 10px; margin: 5px 0;">
                    <strong>🤖 الخبير ({exchange['response_time']:.1f}s):</strong><br>
                    {exchange['answer'][:500]}{'...' if len(exchange['answer']) > 500 else ''}
                </div>
                """, unsafe_allow_html=True)
    
    st.divider()
    
    with st.expander("💡 أسئلة سريعة (إجابة فورية)", expanded=True):
        quick_questions = [
            "ما هي الألوان الأساسية؟",
            "اشرح جين الانتشار",
            "ما هي الوراثة المرتبطة بالجنس؟",
            "اشرح أنماط الريش",
            "ما هو التخفيف؟",
            "اشرح الأحمر المتنحي"
        ]
        
        cols = st.columns(3)
        for i, question in enumerate(quick_questions):
            col_idx = i % 3
            if cols[col_idx].button(f"⚡ {question}", key=f"quick_{i}", use_container_width=True):
                st.session_state['current_question'] = question
    
    current_question = st.session_state.get('current_question', '')
    user_input = st.text_area(
        "اكتب سؤالك هنا:",
        value=current_question,
        height=100,
        placeholder="مثال: كيف أحصل على حمام أزرق موحد؟"
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ask_button = st.button("🚀 اسأل الآن", type="primary", use_container_width=True)
    
    with col2:
        clear_input = st.button("🗑️ مسح السؤال", use_container_width=True)
        if clear_input:
            st.session_state['current_question'] = ''
            st.rerun()
    
    with col3:
        clear_chat = st.button("🔄 مسح المحادثة", use_container_width=True)
        if clear_chat:
            st.session_state.chat_history = []
            st.rerun()
    
    with col4:
        show_tips = st.button("💡 نصائح السرعة", use_container_width=True)
    
    if show_tips:
        with st.expander("⚡ نصائح للحصول على أسرع إجابة", expanded=True):
            st.info("""
            **🚀 للحصول على إجابة فورية (أقل من ثانية):**
            - استخدم الأسئلة المقترحة أعلاه
            - اسأل عن: الألوان الأساسية، الأنماط، الوراثة المرتبطة بالجنس
            
            **⚡ للإجابات السريعة (1-3 ثواني):**
            - اسأل أسئلة واضحة ومحددة
            - استخدم مصطلحات مثل: جين، وراثة، لون، نمط
            
            **🎯 أمثلة على أسئلة سريعة:**
            - "كيف أحصل على حمام أحمر؟"
            - "ما الفرق بين البار والتشيكر؟"
            - "كيف يورث اللون الأزرق؟"
            """)
    
    if ask_button and user_input.strip():
        start_time = time.time()
        
        with st.spinner("🔍 جاري البحث..."):
            instant_answer = fast_ai.get_instant_answer(user_input)
            
            if instant_answer:
                response_time = time.time() - start_time
                memory.add_exchange(user_input, instant_answer, response_time)
                
                st.success(f"⚡ **إجابة فورية ({response_time:.2f}s):**")
                st.markdown(instant_answer)
                
                st.session_state['current_question'] = ''
                st.rerun()
                
            else:
                context = memory.get_context()
                smart_answer = fast_ai.get_smart_answer(user_input, context)
                
                response_time = time.time() - start_time
                memory.add_exchange(user_input, smart_answer, response_time)
                
                if response_time < 3:
                    st.success(f"🚀 **إجابة سريعة ({response_time:.2f}s):**")
                elif response_time < 6:
                    st.info(f"⚡ **إجابة ({response_time:.2f}s):**")
                else:
                    st.warning(f"🐌 **إجابة بطيئة ({response_time:.2f}s):**")
                
                st.markdown(smart_answer)
                
                st.session_state['current_question'] = ''
                st.rerun()

with tab2:
    st.header("🧬 الحاسبة الوراثية السريعة")
    
    parent_inputs = {'male': {}, 'female': {}}
    input_col, result_col = st.columns([2, 3])
    
    with input_col:
        st.subheader("📝 إدخال البيانات")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**♂️ الذكر (الأب)**")
            for gene, data in GENE_DATA.items():
                with st.expander(f"{data['display_name_ar']}"):
                    choices = ["(لا اختيار)"] + [v['name'] for v in data['alleles'].values()]
                    parent_inputs['male'][f'{gene}_visible'] = st.selectbox(
                        "الصفة الظاهرية", choices, key=f"male_{gene}_visible"
                    )
                    parent_inputs['male'][f'{gene}_hidden'] = st.selectbox(
                        "الصفة الخفية", choices, key=f"male_{gene}_hidden"
                    )
        
        with col2:
            st.write("**♀️ الأنثى (الأم)**")
            for gene, data in GENE_DATA.items():
                with st.expander(f"{data['display_name_ar']}"):
                    choices = ["(لا اختيار)"] + [v['name'] for v in data['alleles'].values()]
                    parent_inputs['female'][f'{gene}_visible'] = st.selectbox(
                        "الصفة الظاهرية", choices, key=f"female_{gene}_visible"
                    )
                    if data['type_en'] != 'sex-linked':
                        parent_inputs['female'][f'{gene}_hidden'] = st.selectbox(
                            "الصفة الخفية", choices, key=f"female_{gene}_hidden"
                        )
                    else:
                        st.info("لا يوجد صفة خفية (مرتبط بالجنس)")
                        parent_inputs['female'][f'{gene}_hidden'] = parent_inputs['female'][f'{gene}_visible']
    
    with result_col:
        st.subheader("📊 النتائج المتوقعة")
        
        if st.button("⚡ احسب الآن", use_container_width=True, type="primary"):
            if not all([
                parent_inputs['male'].get('B_visible') != "(لا اختيار)",
                parent_inputs['female'].get('B_visible') != "(لا اختيار)"
            ]):
                st.error("⚠️ الرجاء اختيار اللون الأساسي لكلا الوالدين.")
            else:
                calc_start = time.time()
                
                with st.spinner("🧮 جاري الحساب..."):
                    results = predict_genetics_final(parent_inputs)
                    calc_time = time.time() - calc_start
                    total = sum(results.values())
                    
                    st.success(f"✅ تم حساب {total} تركيبة في {calc_time:.2f} ثانية!")
                    st.session_state.session_stats['calculations'] += 1
                    
                    df_results = pd.DataFrame([
                        {
                            'النمط الظاهري': phenotype,
                            'النمط الوراثي': genotype,
                            'العدد': count,
                            'النسبة %': f"{(count/total)*100:.1f}%"
                        }
                        for (phenotype, genotype), count in results.items()
                    ])
                    
                    st.dataframe(df_results, use_container_width=True)
                    
                    chart_data = df_results.set_index('النمط الظاهري')['العدد']
                    st.bar_chart(chart_data)
                    
                    with st.expander("📈 تحليل سريع للنتائج", expanded=False):
                        st.write("**التوزيع:**")
                        for _, row in df_results.iterrows():
                            st.write(f"• {row['النمط الظاهري']}: {row['النسبة %']}")

with tab3:
    st.header("📊 إحصائيات الأداء المباشرة")
    
    if st.session_state.chat_history:
        response_times = [ex['response_time'] for ex in st.session_state.chat_history]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_response = sum(response_times) / len(response_times)
            st.metric("⚡ متوسط الاستجابة", f"{avg_response:.2f}s")
        
        with col2:
            fastest = min(response_times)
            st.metric("🚀 أسرع إجابة", f"{fastest:.2f}s")
        
        with col3:
            instant_count = len([t for t in response_times if t < 1])
            st.metric("⚡ إجابات فورية", f"{instant_count}")
        
        with col4:
            slow_count = len([t for t in response_times if t > 5])
            st.metric("🐌 إجابات بطيئة", f"{slow_count}")
        
        st.subheader("📈 توزيع أوقات الاستجابة")
        df_times = pd.DataFrame({
            'السؤال': [f"س{i+1}" for i in range(len(response_times))],
            'وقت الاستجابة (ثانية)': response_times
        })
        st.line_chart(df_times.set_index('السؤال'))
        
        st.subheader("🔍 تفاصيل الأداء")
        
        performance_categories = {
            "فوري (< 1s)": [ex for ex in st.session_state.chat_history if ex['response_time'] < 1],
            "سريع (1-3s)": [ex for ex in st.session_state.chat_history if 1 <= ex['response_time'] < 3],
            "عادي (3-5s)": [ex for ex in st.session_state.chat_history if 3 <= ex['response_time'] < 5],
            "بطيء (> 5s)": [ex for ex in st.session_state.chat_history if ex['response_time'] >= 5]
        }
        
        for category, exchanges in performance_categories.items():
            if exchanges:
                with st.expander(f"{category} - {len(exchanges)} سؤال"):
                    for ex in exchanges:
                        st.write(f"• **{ex['question'][:50]}...** ({ex['response_time']:.2f}s)")
        
        st.subheader("🎯 تحليل الأداء")
        
        if avg_response < 2:
            st.success("🚀 **أداء ممتاز!** متوسط الاستجابة أقل من ثانيتين")
        elif avg_response < 4:
            st.info("⚡ **أداء جيد!** متوسط الاستجابة أقل من 4 ثوان")
        else:
            st.warning("🐌 **يمكن تحسين الأداء** - جرب الأسئلة المقترحة للإجابات الفورية")
        
        with st.expander("💡 نصائح تحسين الأداء"):
            st.write("""
            **لتحسين سرعة الاستجابة:**
            
            1. **استخدم الأسئلة المقترحة** - إجابة فورية
            2. **اسأل أسئلة محددة** - تجنب الأسئلة المعقدة
            3. **استخدم المصطلحات الصحيحة** - مثل "جين"، "وراثة"، "لون"
            4. **جرب الحاسبة** - للحسابات العملية السريعة
            
            **أمثلة على أسئلة سريعة:**
            • ما هي الألوان الأساسية؟
            • كيف يعمل جين الانتشار؟
            • ما الفرق بين الذكر والأنثى في الوراثة؟
            """)
    
    else:
        st.info("📊 لا توجد بيانات أداء بعد. ابدأ بطرح بعض الأسئلة لرؤية الإحصائيات!")
        
        st.subheader("ℹ️ معلومات النظام")
        
        system_info = {
            "🤖 محرك الذكاء": "Google Gemini 1.5 Flash",
            "⚡ نوع الاستجابة": "مباشرة بدون Langchain",
            "🧠 قاعدة المعرفة": "محلية + ذكية",
            "💾 الذاكرة": "15 محادثة أخيرة",
            "🚀 الهدف": "استجابة أقل من 3 ثوان"
        }
        
        for key, value in system_info.items():
            st.write(f"**{key}**: {value}")

# --- 9. شريط جانبي محسن ---
with st.sidebar:
    st.header("🔧 أدوات سريعة")
    
    st.subheader("📊 حالة النظام")
    if fast_ai.model:
        st.success("🟢 جاهز للاستخدام")
        st.write("🚀 **سرعة متوقعة:**")
        st.write("• إجابات فورية: < 1s")
        st.write("• إجابات ذكية: 1-3s")
    else:
        st.error("🔴 غير متاح")
        st.write(f"السبب: {fast_ai.error}")
    
    with st.expander("📚 دليل الجينات السريع"):
        st.write("**🎨 الألوان الأساسية:**")
        st.write("• BA = آش ريد (سائد)")
        st.write("• + = أزرق/أسود (طبيعي)")
        st.write("• b = بني (متنحي)")
        
        st.write("**🪶 الأنماط:**")
        st.write("• CT = مخملي (أقوى)")
        st.write("• C = تشيكر")
        st.write("• + = بار (طبيعي)")
        st.write("• c = بدون نمط (متنحي)")
        
        st.write("**⚡ جينات أخرى:**")
        st.write("• S = انتشار (سائد)")
        st.write("• d = تخفيف (متنحي)")
        st.write("• e = أحمر متنحي")
    
    st.subheader("⚡ اختصارات سريعة")
    
    if st.button("🎨 الألوان الأساسية", use_container_width=True):
        st.session_state['current_question'] = "ما هي الألوان الأساسية؟"
        st.rerun()
    
    if st.button("🪶 أنماط الريش", use_container_width=True):
        st.session_state['current_question'] = "اشرح أنماط الريش"
        st.rerun()
    
    if st.button("♂️♀️ الوراثة الجنسية", use_container_width=True):
        st.session_state['current_question'] = "ما هي الوراثة المرتبطة بالجنس؟"
        st.rerun()
    
    if st.button("🔸 جين الانتشار", use_container_width=True):
        st.session_state['current_question'] = "اشرح جين الانتشار"
        st.rerun()
    
    st.divider()
    if st.button("🔄 إعادة تعيين كاملة", type="secondary"):
        for key in ['chat_history', 'session_stats', 'current_question']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# --- 10. تذييل محسن ---
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    session_duration = datetime.now() - st.session_state.session_stats['start_time']
    st.metric("⏰ مدة الجلسة", f"{int(session_duration.total_seconds() / 60)} دقيقة")

with col2:
    if st.session_state.chat_history:
        avg_time = sum([ex['response_time'] for ex in st.session_state.chat_history]) / len(st.session_state.chat_history)
        st.metric("⚡ متوسط السرعة", f"{avg_time:.1f}s")
    else:
        st.metric("⚡ متوسط السرعة", "-- s")

with col3:
    total_interactions = st.session_state.session_stats['questions'] + st.session_state.session_stats['calculations']
    st.metric("📊 إجمالي التفاعلات", total_interactions)

st.markdown("""
<div style='text-align: center; color: #666; margin-top: 20px;'>
    <p>🚀 <strong>العرّاب للجينات - السرعة الفائقة</strong></p>
    <p>⚡ محسن للسرعة مثل ChatGPT | 🧬 خبير في وراثة الحمام الزاجل</p>
    <p style='font-size: 12px;'>النسخة Ultra Fast - بدون Langchain | استجابة مباشرة من Gemini</p>
</div>
""", unsafe_allow_html=True)
