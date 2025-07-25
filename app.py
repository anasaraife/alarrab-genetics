# ===================================================================
# 🚀 العرّاب للجينات V54.0 - الوكيل الموثوق
# تم إعادة تصميم الوكيل ليعتمد حصراً على قاعدة المعرفة المحلية (RAG)
# ===================================================================

import streamlit as st
from itertools import product
import collections
import pandas as pd
import google.generativeai as genai
import time
from datetime import datetime

# --- 1. إعدادات الصفحة ---
st.set_page_config(layout="wide", page_title="العرّاب للجينات - الوكيل الموثوق")

# --- 2. قاعدة البيانات الوراثية ---
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

# --- 3. المحرك الوراثي (بدون تغيير) ---
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
            phenotypes['B'] = 'أحمر متنحي'; phenotypes['C'] = ''
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
    # ... (الكود الكامل لهذه الوظيفة موجود في النسخ السابقة، تم إخفاؤه هنا للاختصار)
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
    'الألوان الأساسية': """... (المحتوى كما هو في النسخة السابقة) ...""",
    'جين الانتشار': """... (المحتوى كما هو في النسخة السابقة) ...""",
    'الوراثة المرتبطة بالجنس': """... (المحتوى كما هو في النسخة السابقة) ...""",
    'أنماط الريش': """... (المحتوى كما هو في النسخة السابقة) ...""",
    'التخفيف': """... (المحتوى كما هو في النسخة السابقة) ...""",
    'أحمر متنحي': """... (المحتوى كما هو في النسخة السابقة) ..."""
}
# إضافة محتوى المعرفة الكامل هنا لتجنب الحذف
ULTRA_FAST_KNOWLEDGE = {
    'الألوان الأساسية': """
🎨 **الألوان الأساسية في الحمام الزاجل:**

**1. الآش ريد (Ash Red) - BA:**
- لون أحمر مائل للرمادي.
- سائد على جميع الألوان الأخرى.
- رمز الجين: BA

**2. الأزرق/أسود (Blue/Black) - +:**
- اللون الطبيعي الأساسي للحمام البري.
- متوسط السيادة.
- رمز الجين: +

**3. البني (Brown/Red) - b:**
- لون بني محمر أو شوكولاتي.
- أكثر الألوان تنحياً.
- رمز الجين: b

**الوراثة:** هذه الألوان مرتبطة بالجنس. **ترتيب السيادة:** BA > + > b
    """,
    'جين الانتشار': """
🔸 **جين الانتشار (Spread Gene - S):**
- ينشر اللون الأساسي على كامل الريشة ويخفي الأنماط.
- جين جسمي (autosomal) سائد.
- **مثال:** حمامة زرقاء + جين الانتشار = زرقاء موحدة (سوداء).
    """,
    'الوراثة المرتبطة بالجنس': """
♂️♀️ **الوراثة المرتبطة بالجنس في الحمام:**
- **الذكور: ZZ** (لديهم نسختان من الجين ويمكن أن يكونوا حاملين).
- **الإناث: ZW** (لديهن نسخة واحدة ولا يمكن أن يكن حاملات).
- **الجينات المرتبطة بالجنس:** اللون الأساسي (B) والتخفيف (d).
    """,
    'أنماط الريش': """
🪶 **أنماط الريش (Pattern Gene - C):**
- جين جسمي (autosomal).
- **ترتيب السيادة:** نمط T المخملي (CT) > التشيكر (C) > البار (شريط) (+) > بدون نمط (c).
    """,
    'التخفيف': """
💧 **جين التخفيف (Dilution Gene - d):**
- يخفف كثافة اللون الأساسي.
- جين متنحي مرتبط بالجنس.
- **التأثير:** أزرق مخفف = فضي، آش ريد مخفف = أصفر.
    """,
    'أحمر متنحي': """
🔴 **الأحمر المتنحي (Recessive Red - e):**
- جين جسمي متنحي قوي.
- يخفي جميع الألوان والأنماط الأساسية الأخرى ويعطي لوناً أحمر موحداً.
- يتطلب نسختين (e/e) ليظهر.
    """
}

# --- 5. الوكيل الموثوق (Reliable Agent) ---
class ReliableAgent:
    def __init__(self):
        if "GEMINI_API_KEY" not in st.secrets:
            self.model = None
            self.error = "مفتاح API غير موجود"
        else:
            try:
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                self.model = genai.GenerativeModel('gemini-1.5-flash',
                    generation_config={"temperature": 0.1, "max_output_tokens": 1000})
                self.error = None
            except Exception as e:
                self.model = None
                self.error = str(e)

    def find_relevant_context(self, query):
        """
        يبحث عن الموضوع الأكثر صلة في قاعدة المعرفة المحلية.
        """
        query_lower = query.lower().strip()
        keywords_map = {
            'الألوان الأساسية': ['لون', 'الوان', 'أساسي', 'اساسي', 'آش ريد', 'أزرق', 'بني', 'ash', 'blue', 'brown'],
            'جين الانتشار': ['انتشار', 'سبريد', 'spread', 'منتشر', 'موحد'],
            'الوراثة المرتبطة بالجنس': ['جنس', 'sex', 'ذكر', 'أنثى', 'zw', 'zz'],
            'أنماط الريش': ['نمط', 'انماط', 'pattern', 'بار', 'تشيكر', 'checker', 'bar', 'مخملي', 'velvet'],
            'التخفيف': ['تخفيف', 'مخفف', 'dilution', 'dilute', 'فاتح', 'باهت', 'فضي', 'أصفر'],
            'أحمر متنحي': ['أحمر متنحي', 'احمر متنحي', 'recessive red']
        }
        
        for topic, keywords in keywords_map.items():
            if any(keyword in query_lower for keyword in keywords):
                return ULTRA_FAST_KNOWLEDGE[topic]
        return None

    def get_grounded_answer(self, query):
        """
        ينتج إجابة موثوقة بناءً على قاعدة المعرفة فقط.
        """
        if not self.model:
            return f"❌ خطأ في النظام: {self.error}"

        context = self.find_relevant_context(query)

        if not context:
            return "لم أجد معلومات دقيقة حول هذا السؤال في قاعدة المعرفة الحالية. هل يمكنك إعادة صياغة السؤال أو طرح سؤال حول أحد المواضيع الأساسية؟"

        try:
            prompt = f"""
            أنت "العرّاب الذكي"، خبير في وراثة الحمام. مهمتك هي الإجابة على سؤال المستخدم بناءً على المعلومات المتوفرة في "السياق" فقط.

            **السياق:**
            ---
            {context}
            ---

            **سؤال المستخدم:** {query}

            **تعليمات:**
            1. أجب باللغة العربية.
            2. لخص الإجابة من السياق بشكل واضح ومباشر.
            3. لا تضف أي معلومات غير موجودة في السياق.
            4. إذا كان السياق لا يجيب على السؤال بشكل مباشر، قل "المعلومات المتوفرة في قاعدة المعرفة لا تجيب على هذا السؤال بدقة."

            **الإجابة:**
            """
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"⚠️ خطأ مؤقت أثناء معالجة الطلب: {str(e)[:100]}..."

# --- 6. إنشاء كائنات النظام ---
agent = ReliableAgent()

# --- 7. واجهة التطبيق ---
st.title("🚀 العرّاب للجينات - الوكيل الموثوق")

tab1, tab2 = st.tabs(["🤖 المحادثة الموثوقة", "🧬 الحاسبة الوراثية"])

with tab1:
    st.header("💬 تحدث مع الخبير")
    st.info("يطرح هذا الوكيل إجاباته بناءً على قاعدة معرفة محلية لضمان الدقة والسرعة.")
    
    # عرض المحادثة
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # استقبال مدخلات المستخدم
    if prompt := st.chat_input("اسأل عن الألوان، الأنماط، أو الوراثة..."):
        # إضافة رسالة المستخدم إلى سجل المحادثة
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # الحصول على إجابة الوكيل وعرضها
        with st.chat_message("assistant"):
            with st.spinner("الخبير يفكر..."):
                response = agent.get_grounded_answer(prompt)
                st.markdown(response)
        
        # إضافة إجابة الوكيل إلى السجل
        st.session_state.messages.append({"role": "assistant", "content": response})

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
                    parent_inputs['male'][f'{gene}_visible'] = st.selectbox("الصفة الظاهرية", choices, key=f"male_{gene}_visible")
                    parent_inputs['male'][f'{gene}_hidden'] = st.selectbox("الصفة الخفية", choices, key=f"male_{gene}_hidden")
        with col2:
            st.write("**♀️ الأنثى (الأم)**")
            for gene, data in GENE_DATA.items():
                with st.expander(f"{data['display_name_ar']}"):
                    choices = ["(لا اختيار)"] + [v['name'] for v in data['alleles'].values()]
                    parent_inputs['female'][f'{gene}_visible'] = st.selectbox("الصفة الظاهرية", choices, key=f"female_{gene}_visible")
                    if data['type_en'] != 'sex-linked':
                        parent_inputs['female'][f'{gene}_hidden'] = st.selectbox("الصفة الخفية", choices, key=f"female_{gene}_hidden")
                    else:
                        st.info("لا يوجد صفة خفية")
                        parent_inputs['female'][f'{gene}_hidden'] = parent_inputs['female'][f'{gene}_visible']
    with result_col:
        st.subheader("📊 النتائج المتوقعة")
        if st.button("⚡ احسب الآن", use_container_width=True, type="primary"):
            if not all([parent_inputs['male'].get('B_visible') != "(لا اختيار)", parent_inputs['female'].get('B_visible') != "(لا اختيار)"]):
                st.error("⚠️ الرجاء اختيار اللون الأساسي لكلا الوالدين.")
            else:
                with st.spinner("🧮 جاري الحساب..."):
                    results = predict_genetics_final(parent_inputs)
                    total = sum(results.values())
                    st.success(f"✅ تم حساب {total} تركيبة!")
                    df_results = pd.DataFrame([{'النمط الظاهري': p, 'النمط الوراثي': g, 'النسبة %': f"{(c/total)*100:.1f}%"} for (p, g), c in results.items()])
                    st.dataframe(df_results, use_container_width=True)
                    chart_data = df_results.set_index('النمط الظاهري')['النسبة %'].str.rstrip('%').astype('float')
                    st.bar_chart(chart_data)
