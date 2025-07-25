# ===================================================================
# 🕊️ العرّاب للجينات V45.0 - النسخة النهائية والمستقرة
# تم تطبيق الحل الاحترافي للمساعد الذكي باستخدام قائمة نماذج ديناميكية
# ===================================================================

import streamlit as st
from itertools import product
import collections
import pandas as pd
import google.generativeai as genai
import json

# --- 1. إعدادات الصفحة ---
st.set_page_config(layout="wide", page_title="العرّاب للجينات")

# --- 2. قاعدة البيانات الوراثية الكاملة ---
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
    },
    'Gr': {
        'display_name_ar': "الجريزل", 'type_en': 'autosomal',
        'alleles': {
            'Gr': {'name': 'جريزل', 'is_recessive': False},
            '+': {'name': 'عادي (غير جريزل)', 'is_recessive': False}
        },
        'dominance': ['Gr', '+']
    },
    'Op': {
        'display_name_ar': "الأوبال", 'type_en': 'autosomal',
        'alleles': {
            '+': {'name': 'عادي (غير أوبال)', 'is_recessive': False},
            'Op': {'name': 'أوبال', 'is_recessive': True}
        },
        'dominance': ['+', 'Op']
    },
    'My': {
        'display_name_ar': "الملكي", 'type_en': 'autosomal',
        'alleles': {
            '+': {'name': 'عادي (غير ملكي)', 'is_recessive': False},
            'My': {'name': 'ملكي', 'is_recessive': True}
        },
        'dominance': ['+', 'My']
    },
    'In': {
        'display_name_ar': "الإنديغو", 'type_en': 'autosomal',
        'alleles': {
            'In': {'name': 'إنديغو', 'is_recessive': False},
            '+': {'name': 'عادي (غير إنديغو)', 'is_recessive': False}
        },
        'dominance': ['In', '+']
    }
}
GENE_ORDER = list(GENE_DATA.keys())
NAME_TO_SYMBOL_MAP = {
    gene: {info['name']: symbol for symbol, info in data['alleles'].items()}
    for gene, data in GENE_DATA.items()
}

# --- 3. المحرك الوراثي والوظائف ---
class GeneticCalculator:
    def describe_phenotype(self, genotype_dict):
        phenotypes = {gene: "" for gene in GENE_ORDER}
        for gene_name, gt_part in genotype_dict.items():
            alleles = gt_part.replace('•//', '').replace('//', '')
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
        for gene in GENE_ORDER:
            phenotype_name = phenotypes.get(gene)
            if gene not in ['B', 'C', 'S', 'e'] and phenotype_name and "عادي" not in phenotype_name:
                desc_parts.append(phenotype_name)
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

def generate_breeding_plan(target_inputs):
    target_genotype = {}
    for gene, phenotype_name in target_inputs.items():
        if phenotype_name and phenotype_name != "(لا اختيار)":
            target_symbol = NAME_TO_SYMBOL_MAP[gene].get(phenotype_name)
            if target_symbol:
                target_genotype[gene] = target_symbol
    if not target_genotype:
        return "⚠️ الرجاء تحديد صفة واحدة على الأقل كهدف للإنتاج."
    target_name_parts = [GENE_DATA[gene]['alleles'][allele]['name'] for gene, allele in target_genotype.items()]
    target_full_name = " ".join(target_name_parts)
    plan = f"### 📝 خطة مقترحة لإنتاج '{target_full_name}'\n\n"
    recessive_genes, dominant_genes = [], []
    for gene, allele in target_genotype.items():
        if GENE_DATA[gene]['alleles'][allele]['is_recessive']:
            recessive_genes.append(gene)
        else:
            dominant_genes.append(gene)
    step = 1
    if dominant_genes:
        plan += f"#### **الخطوة {step}: إدخال الصفات السائدة**\n"
        plan += "الصفات التالية **سائدة**. يكفي أن يكون أحد الأبوين يحملها لإنتاجها:\n"
        for gene in dominant_genes:
            plan += f"- **{GENE_DATA[gene]['display_name_ar']}** ({GENE_DATA[gene]['alleles'][target_genotype[gene]]['name']})\n"
        plan += "\n**التوصية:** قم بتزويج طائر يظهر عليه هذه الصفات مع أفضل طيورك.\n\n---\n"
        step += 1
    if recessive_genes:
        plan += f"#### **الخطوة {step}: إنتاج الصفات المتنحية (خطة من جيلين)**\n"
        plan += "الصفات التالية **متنحية** وتتطلب خطة من جيلين لإظهارها:\n"
        plan += "**الجيل الأول (F1): إنتاج الحَمَلة (Carriers)**\n"
        plan += "1.  اختر طائرًا نقيًا لكل صفة متنحية مطلوبة:\n"
        for gene in recessive_genes:
            allele = target_genotype[gene]
            plan += f"    - طائر **{GENE_DATA[gene]['alleles'][allele]['name']}** (`{allele}//{allele}`)\n"
        plan += "2.  قم بتزويج هذه الطيور مع طيور نقية عادية (Wild Type).\n"
        plan += "**النتيجة (F1):** كل الإنتاج سيكون عادي المظهر ولكنه **حامل للصفات المتنحية** بشكل خفي.\n\n"
        plan += "**الجيل الثاني (F2): إظهار الهدف**\n"
        plan += "1. قم بتزويج الأبناء الحاملين للصفات من الجيل الأول مع بعضهم البعض.\n"
        plan += "**النتيجة (F2):** ستظهر الصفات المتنحية المطلوبة في جزء من النسل (حوالي 25% لكل صفة).\n"
    return plan

# --- 4. وظائف المساعد الذكي (Agent) - الحل الاحترافي ---
def get_gemini_response(query):
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        
        available_genes = "، ".join([f"{data['display_name_ar']} ({', '.join([a['name'] for a in data['alleles'].values()])})" for data in GENE_DATA.values()])

        prompt = f"""
        أنت "العرّاب الذكي"، خبير في وراثة الحمام. مهمتك هي تحليل سؤال المستخدم عن تهجين الحمام واستخراج التركيب الجيني للأب والأم.
        
        الجينات المتاحة هي: {available_genes}

        سؤال المستخدم: "{query}"

        المطلوب: قم بإرجاع رد بصيغة JSON فقط، بدون أي نص إضافي. يجب أن يحتوي الـ JSON على مفتاحين: "male" و "female".
        كل مفتاح يجب أن يحتوي على قاموس للجينات. لكل جين، حدد "visible" (الصفة الظاهرية) و "hidden" (الصفة المحمولة).
        إذا لم يذكر المستخدم صفة محمولة، اجعل قيمة "hidden" نفس قيمة "visible".
        إذا لم يذكر المستخدم جيناً معيناً، أهمله من القاموس.

        مثال على المخرج المطلوب:
        {{
          "male": {{
            "B_visible": "أزرق/أسود",
            "B_hidden": "بني"
          }},
          "female": {{
            "B_visible": "آش ريد",
            "B_hidden": "آش ريد"
          }}
        }}
        """
        
        # قائمة النماذج المتاحة بترتيب الأولوية
        available_models = [
            'gemini-2.5-flash',
            'gemini-2.0-flash',
            'gemini-2.5-flash-lite',
            'gemini-2.0-flash-lite'
        ]
        
        response = None
        used_model = None
        
        # جرب النماذج بالترتيب حتى تجد واحدًا يعمل
        for model_name in available_models:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                used_model = model_name
                st.success(f"✅ تم استخدام نموذج: {used_model}")
                break
            except Exception:
                st.info(f"تعذر استخدام نموذج {model_name}، جاري تجربة النموذج التالي...")
                continue
        
        if response:
            return response.text
        else:
            raise Exception("لم يتمكن من العثور على أي نموذج متاح.")
            
    except Exception as e:
        st.error(f"حدث خطأ أثناء التواصل مع المساعد الذكي: {e}")
        st.info("يرجى التأكد من صحة مفتاح Gemini API في أسرار التطبيق (Secrets).")
        return None

# --- 5. واجهة التطبيق ---
st.title("🕊️ العرّاب للجينات (V45 - النسخة النهائية)")

def clear_all_inputs():
    for key in st.session_state.keys():
        if key.startswith("male_") or key.startswith("female_") or key.startswith("target_"):
            st.session_state[key] = "(لا اختيار)"

tab1, tab2, tab3 = st.tabs(["🧬 الحاسبة الذكية", "🎯 مخطط الإنتاج", "🤖 المساعد الذكي (Agent)"])

with tab1:
    parent_inputs = {'male': {}, 'female': {}}
    input_col, result_col = st.columns([2, 3])
    with input_col:
        st.header("📝 المدخلات")
        st.button("🔄 مسح كل الخيارات", on_click=clear_all_inputs, use_container_width=True, key="clear_tab1")
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("♂️ الذكر (الأب)")
            for gene, data in GENE_DATA.items():
                with st.expander(f"{data['display_name_ar']}"):
                    choices = ["(لا اختيار)"] + [v['name'] for v in data['alleles'].values()]
                    parent_inputs['male'][f'{gene}_visible'] = st.selectbox("الصفة الظاهرية", choices, key=f"male_{gene}_visible")
                    parent_inputs['male'][f'{gene}_hidden'] = st.selectbox("الصفة الخفية", choices, key=f"male_{gene}_hidden")
        with col2:
            st.subheader("♀️ الأنثى (الأم)")
            for gene, data in GENE_DATA.items():
                with st.expander(f"{data['display_name_ar']}"):
                    choices = ["(لا اختيار)"] + [v['name'] for v in data['alleles'].values()]
                    parent_inputs['female'][f'{gene}_visible'] = st.selectbox("الصفة الظاهرية", choices, key=f"female_{gene}_visible")
                    if data['type_en'] != 'sex-linked':
                        parent_inputs['female'][f'{gene}_hidden'] = st.selectbox("الصفة الخفية", choices, key=f"female_{gene}_hidden")
                    else:
                        st.info("لا يوجد صفة خفية (مرتبط بالجنس)")
                        parent_inputs['female'][f'{gene}_hidden'] = parent_inputs['female'][f'{gene}_visible']
    with result_col:
        st.header("📊 النتائج")
        if st.button("احسب النتائج", use_container_width=True, type="primary"):
            if not all([parent_inputs['male'].get('B_visible') != "(لا اختيار)", parent_inputs['female'].get('B_visible') != "(لا اختيار)"]):
                st.error("⚠️ الرجاء اختيار اللون الأساسي لكلا الوالدين.")
            else:
                with st.spinner("جاري حساب الاحتمالات..."):
                    results = predict_genetics_final(parent_inputs)
                    total = sum(results.values())
                    st.success(f"تم حساب {total} تركيبة محتملة بنجاح!")
                    st.subheader("قائمة النتائج:")
                    chart_data = []
                    for (phenotype, genotype), count in sorted(results.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / total) * 100
                        st.write(f"- **{percentage:.2f}%** - {phenotype} `{genotype}`")
                        chart_data.append({'التركيب المحتمل': f"{phenotype} ({genotype})", 'الاحتمالية': percentage})
                    if chart_data:
                        st.subheader("الرسم البياني للنتائج:")
                        df = pd.DataFrame(chart_data)
                        st.bar_chart(df.set_index('التركيب المحتمل'))

with tab2:
    st.header("🎯 المخطط العكسي لإنتاج الصفات")
    st.write("اختر الصفات التي تريد إنتاجها في الطائر الهدف، وسيقوم التطبيق بوضع خطة عمل مقترحة.")
    target_inputs = {}
    cols = st.columns(3)
    col_idx = 0
    for gene, data in GENE_DATA.items():
        with cols[col_idx]:
            choices = ["(لا اختيار)"] + [v['name'] for v in data['alleles'].values()]
            target_inputs[gene] = st.selectbox(f"اختر {data['display_name_ar']}", choices, key=f"target_{gene}")
        col_idx = (col_idx + 1) % 3
    if st.button("📝 ضع الخطة", use_container_width=True, type="primary"):
        with st.spinner("جاري إعداد الخطة..."):
            plan = generate_breeding_plan(target_inputs)
            st.markdown(plan)

with tab3:
    st.header("🤖 تحدث مع العرّاب الذكي (Agent)")
    st.write("اطرح سؤالك عن التهجين بلغة طبيعية، وسيقوم الوكيل الذكي بتحليله وحساب النتائج لك.")
    user_query = st.text_area("مثال: ما هو ناتج تزاوج ذكر أزرق بار حامل للبني مع أنثى آش ريد؟", height=100)
    if st.button("اسأل الوكيل الذكي", use_container_width=True, type="primary"):
        if not user_query:
            st.warning("الرجاء إدخال سؤالك.")
        else:
            with st.spinner("الوكيل الذكي يفكر... 🤔"):
                json_response_str = get_gemini_response(user_query)
                if json_response_str:
                    try:
                        clean_json_str = json_response_str.strip().replace("```json", "").replace("```", "").strip()
                        extracted_data = json.loads(clean_json_str)
                        st.subheader("تحليل الوكيل الذكي لسؤالك:")
                        st.json(extracted_data)
                        agent_parent_inputs = {'male': {}, 'female': {}}
                        for parent, genes in extracted_data.items():
                            for key, value in genes.items():
                                agent_parent_inputs[parent][key] = value
                        st.subheader("📊 نتائج التهجين بناءً على تحليل الوكيل:")
                        results = predict_genetics_final(agent_parent_inputs)
                        total = sum(results.values())
                        st.success(f"تم حساب {total} تركيبة محتملة بنجاح!")
                        chart_data = []
                        for (phenotype, genotype), count in sorted(results.items(), key=lambda x: x[1], reverse=True):
                            percentage = (count / total) * 100
                            st.write(f"- **{percentage:.2f}%** - {phenotype} `{genotype}`")
                            chart_data.append({'التركيب المحتمل': f"{phenotype} ({genotype})", 'الاحتمالية': percentage})
                        if chart_data:
                            df = pd.DataFrame(chart_data)
                            st.bar_chart(df.set_index('التركيب المحتمل'))
                    except (json.JSONDecodeError, KeyError) as e:
                        st.error(f"لم يتمكن الوكيل الذكي من فهم السؤال بشكل صحيح. حاول صياغة السؤال بشكل أوضح.")
                        st.write("الاستجابة المستلمة:")
                        st.write(json_response_str)
